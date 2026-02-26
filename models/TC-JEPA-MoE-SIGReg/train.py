"""
Train JEPA MoE + SIGReg text classifier (always-on SIGReg).
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent.parent
_model_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_model_dir))

from dataset.banking77 import Banking77Dataset, get_labels as get_banking77_labels, load_banking77_dataset
from dataset.clinc_oos import CLINCOOSDataset, get_labels as get_clinc_labels, load_clinc_oos_dataset
from dataset.yahoo_answers import YahooAnswersDataset, get_labels as get_yahoo_labels, load_yahoo_answers_dataset
from encoders.OpenCLIP import OpenCLIPTextEncoder
from metrics.moe import conditional_routing_entropy, expert_usage_entropy
from metrics.representation import covariance_spectrum, effective_rank, variance_ratio
from utils.embedding_cache import build_cached_loaders, get_or_build_text_embedding_cache
from utils.losses import build_sigreg_loss_fn, compute_sigreg_bcs_loss, cosine_similarity_loss
from utils.train import save_checkpoint, setup_device, setup_seed

from model import MoESigRegJEPATextClassifier


def collate_fn(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels


def train_epoch(model, loader, optimizer, device, label_embeddings, sigreg_loss_fn, sigreg_weight):
    model.train()
    total_loss = 0.0
    total_sigreg = 0.0
    n = 0
    for inputs, labels in tqdm(loader, desc="Train"):
        labels = labels.to(device)
        use_amp = device.type == "cuda"
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            if isinstance(inputs, torch.Tensor):
                input_emb = inputs.to(device=device, dtype=torch.float32)
            else:
                input_emb = model.encode_text(inputs)
            pred_emb = model(input_emb)
            pred_emb = pred_emb / pred_emb.norm(dim=-1, keepdim=True)
            targets = label_embeddings[labels]
            alignment_loss = cosine_similarity_loss(pred_emb, targets)
            sigreg_loss = compute_sigreg_bcs_loss(
                sigreg_loss_fn=sigreg_loss_fn,
                pred_emb=pred_emb,
                targets=targets,
                reference=alignment_loss,
            )
            loss = alignment_loss + sigreg_weight * sigreg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_sigreg += sigreg_loss.item() * labels.size(0)
        n += labels.size(0)
    return total_loss / n if n else 0.0, total_sigreg / n if n else 0.0


@torch.no_grad()
def full_evaluation(model, loader, device, label_embeddings):
    """Single-pass eval: loss, accuracy, representation metrics, and MoE diagnostics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    all_pred = []
    all_target = []
    all_gate_probs = []

    for inputs, labels in tqdm(loader, desc="Eval"):
        labels = labels.to(device)
        if isinstance(inputs, torch.Tensor):
            input_emb = inputs.to(device=device, dtype=torch.float32)
        else:
            input_emb = model.encode_text(inputs)
        pred_emb, gate_probs, _ = model.forward_with_diagnostics(input_emb)
        pred_emb = pred_emb / pred_emb.norm(dim=-1, keepdim=True)
        targets = label_embeddings[labels]

        total_loss += cosine_similarity_loss(pred_emb, targets).item() * labels.size(0)
        logits = pred_emb @ label_embeddings.T
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)
        all_pred.append(pred_emb.cpu())
        all_target.append(targets.cpu())
        all_gate_probs.append(gate_probs.cpu())

    pred_emb_all = torch.cat(all_pred, dim=0)
    target_emb_all = torch.cat(all_target, dim=0)
    gate_probs_all = torch.cat(all_gate_probs, dim=0)
    usage_h, usage_h_norm = expert_usage_entropy(gate_probs_all)
    cond_h, cond_h_norm = conditional_routing_entropy(gate_probs_all)
    return {
        "eval_loss": total_loss / n if n else 0.0,
        "eval_acc": correct / n if n else 0.0,
        "pred_effective_rank": effective_rank(pred_emb_all),
        "target_effective_rank": effective_rank(target_emb_all),
        "variance_ratio": variance_ratio(pred_emb_all, target_emb_all),
        "pred_cov_top1_eig": float(covariance_spectrum(pred_emb_all)[0].item()),
        "target_cov_top1_eig": float(covariance_spectrum(target_emb_all)[0].item()),
        "expert_usage_entropy": usage_h,
        "expert_usage_entropy_norm": usage_h_norm,
        "conditional_routing_entropy": cond_h,
        "conditional_routing_entropy_norm": cond_h_norm,
    }


def write_metrics_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["yahoo_answers", "banking77", "clinc_oos"], default="yahoo_answers")
    parser.add_argument("--clinc_config", type=str, choices=["plus", "small", "imbalanced"], default="plus")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--predictor_hidden", type=int, default=1024)
    parser.add_argument("--moe_num_experts", type=int, default=4)
    parser.add_argument("--sigreg_weight", type=float, default=0.05)
    parser.add_argument("--sigreg_num_slices", type=int, default=256)
    parser.add_argument("--sigreg_lmbd", type=float, default=10.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--metrics_csv", type=str, default=None)
    parser.add_argument("--embedding_cache_dir", type=str, default=None, help="Directory to store/reuse frozen text embeddings")
    parser.add_argument("--precompute_batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = setup_device(args.device)
    setup_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "yahoo_answers":
        ds_dict = load_yahoo_answers_dataset(cache_dir=args.cache_dir)
        labels_list = get_yahoo_labels(cache_dir=args.cache_dir)
        train_ds = YahooAnswersDataset(ds_dict["train"])
        test_ds = YahooAnswersDataset(ds_dict["test"])
    elif args.dataset == "banking77":
        ds_dict = load_banking77_dataset(cache_dir=args.cache_dir)
        labels_list = get_banking77_labels(cache_dir=args.cache_dir)
        train_ds = Banking77Dataset(ds_dict["train"])
        test_ds = Banking77Dataset(ds_dict["test"])
    else:
        ds_dict = load_clinc_oos_dataset(config=args.clinc_config, cache_dir=args.cache_dir)
        labels_list = get_clinc_labels(config=args.clinc_config, cache_dir=args.cache_dir)
        train_ds = CLINCOOSDataset(ds_dict["train"])
        test_ds = CLINCOOSDataset(ds_dict["test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    encoder = OpenCLIPTextEncoder(
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=device,
    )
    with torch.no_grad():
        label_emb = encoder(labels_list)
        label_emb = label_emb / label_emb.norm(dim=-1, keepdim=True)
    label_embeddings = label_emb.to(device)

    if args.embedding_cache_dir:
        dataset_id = args.dataset if args.dataset != "clinc_oos" else f"{args.dataset}_{args.clinc_config}"
        cache_payload = get_or_build_text_embedding_cache(
            cache_dir=args.embedding_cache_dir,
            dataset_id=dataset_id,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            train_ds=train_ds,
            test_ds=test_ds,
            labels_list=labels_list,
            encoder=encoder,
            device=device,
            precompute_batch_size=args.precompute_batch_size,
        )
        del train_ds, test_ds, ds_dict
        encoder.model = encoder.model.cpu()
        import gc; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        train_loader, test_loader = build_cached_loaders(cache_payload, args.batch_size)
        label_embeddings = cache_payload["label_embeddings"].to(device=device, dtype=torch.float32)
        label_embeddings = label_embeddings / label_embeddings.norm(dim=-1, keepdim=True)
        print("Using precomputed frozen encoder embeddings for train/test splits.")

    model = MoESigRegJEPATextClassifier(
        encoder=encoder,
        predictor_hidden_dim=args.predictor_hidden,
        moe_num_experts=args.moe_num_experts,
        device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)
    sigreg_loss_fn = build_sigreg_loss_fn(
        use_sigreg=True,
        num_slices=args.sigreg_num_slices,
        lmbd=args.sigreg_lmbd,
    )
    metrics_rows = []
    metrics_csv_path = Path(args.metrics_csv) if args.metrics_csv else save_dir / "training_metrics_moe_sigreg.csv"

    for ep in range(args.epochs):
        train_loss, train_sigreg = train_epoch(
            model, train_loader, optimizer, device, label_embeddings, sigreg_loss_fn=sigreg_loss_fn, sigreg_weight=args.sigreg_weight
        )
        print(f"Epoch {ep + 1}/{args.epochs} | train_loss={train_loss:.4f} | train_sigreg={train_sigreg:.4f}")
        metrics_rows.append({"epoch": ep + 1, "train_loss": train_loss, "train_sigreg": train_sigreg})

    eval_results = full_evaluation(model, test_loader, device, label_embeddings)
    metrics_rows[-1].update(eval_results)
    save_checkpoint(save_dir / "best_jepa_moe_sigreg.pt", model=model, optimizer=optimizer, epoch=args.epochs, eval_acc=eval_results["eval_acc"])
    print(
        f"Final eval | eval_loss={eval_results['eval_loss']:.4f} | eval_acc={eval_results['eval_acc']:.4f}"
    )
    print(
        f"  repr/moe: pred_erank={eval_results['pred_effective_rank']:.4f} "
        f"| var_ratio={eval_results['variance_ratio']:.4f} "
        f"| usage_h_norm={eval_results['expert_usage_entropy_norm']:.4f}"
    )
    print(f"Final test accuracy: {eval_results['eval_acc']:.4f}")
    write_metrics_csv(metrics_rows, metrics_csv_path)
    print(f"Saved training metrics CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()
