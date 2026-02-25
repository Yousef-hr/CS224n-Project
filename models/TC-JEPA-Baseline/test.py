"""
Test/evaluate JEPA baseline text classifier on Banking77 / CLINC-OOS.
Uses same dataset and evaluation pattern as project test.py (accuracy, per-class, optional repr metrics).
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Project root and model dir on path for imports
_root = Path(__file__).resolve().parent.parent.parent
_model_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_model_dir))

from dataset.banking77 import Banking77Dataset, get_labels as get_banking77_labels, load_banking77_dataset
from dataset.clinc_oos import CLINCOOSDataset, get_labels as get_clinc_labels, load_clinc_oos_dataset
from dataset.yahoo_answers import YahooAnswersDataset, get_labels as get_yahoo_labels, load_yahoo_answers_dataset
from encoders.OpenCLIP import OpenCLIPTextEncoder
from utils.train import setup_device, load_checkpoint
from metrics.representation import covariance_spectrum, effective_rank, variance_ratio

from model import BaselineJEPATextClassifier


def collate_fn(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["yahoo_answers", "banking77", "clinc_oos"], default="yahoo_answers")
    parser.add_argument("--clinc_config", type=str, choices=["plus", "small", "imbalanced"], default="plus")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache dir")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_jepa_baseline.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--predictor_hidden", type=int, default=1024)
    parser.add_argument("--baseline_dropout", type=float, default=0.1)
    parser.add_argument("--report_repr_metrics", action="store_true", help="Compute representation-level metrics")
    parser.add_argument("--repr_topk_eigs", type=int, default=10, help="Top-k eigenvalues to print from covariance spectrum")
    args = parser.parse_args()

    device = setup_device(args.device)
    ckpt_path = Path(args.checkpoint)

    # Data
    if args.dataset == "yahoo_answers":
        ds_dict = load_yahoo_answers_dataset(cache_dir=args.cache_dir)
        labels_list = get_yahoo_labels(cache_dir=args.cache_dir)
        test_ds = YahooAnswersDataset(ds_dict["test"])
    elif args.dataset == "banking77":
        ds_dict = load_banking77_dataset(cache_dir=args.cache_dir)
        labels_list = get_banking77_labels(cache_dir=args.cache_dir)
        test_ds = Banking77Dataset(ds_dict["test"])
    else:
        ds_dict = load_clinc_oos_dataset(config=args.clinc_config, cache_dir=args.cache_dir)
        labels_list = get_clinc_labels(config=args.clinc_config, cache_dir=args.cache_dir)
        test_ds = CLINCOOSDataset(ds_dict["test"])
    idx_to_label = {i: lab for i, lab in enumerate(labels_list)}
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Encoder and label embeddings
    encoder = OpenCLIPTextEncoder(
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=device,
    )
    with torch.no_grad():
        label_embeddings = encoder(labels_list)
        label_embeddings = label_embeddings / label_embeddings.norm(dim=-1, keepdim=True)
    label_embeddings = label_embeddings.to(device)

    # Model
    model = BaselineJEPATextClassifier(
        encoder=encoder,
        predictor_hidden_dim=args.predictor_hidden,
        baseline_dropout=args.baseline_dropout,
        device=device,
    ).to(device)
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    all_pred_emb = []
    all_target_emb = []

    with torch.no_grad():
        for texts, labels in test_loader:
            labels = labels.to(device)
            input_emb = model.encode_text(texts)
            pred_emb = model(input_emb)
            pred_emb = pred_emb / pred_emb.norm(dim=-1, keepdim=True)
            logits = pred_emb @ label_embeddings.T
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            if args.report_repr_metrics:
                target_emb = label_embeddings[labels]
                all_pred_emb.append(pred_emb.cpu())
                all_target_emb.append(target_emb.cpu())

    acc = correct / total
    print(f"Test accuracy: {acc:.4f} ({correct}/{total})")

    # Per-class metrics
    n_classes = len(idx_to_label)
    class_correct = [0] * n_classes
    class_total = [0] * n_classes
    for p, l in zip(all_preds, all_labels):
        class_total[l] += 1
        if p == l:
            class_correct[l] += 1

    print("\nPer-class accuracy (top 15 by support):")
    sorted_by_support = sorted(range(n_classes), key=lambda i: class_total[i], reverse=True)
    for i in sorted_by_support[:15]:
        c_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        print(f"  {idx_to_label[i]:40s}: {c_acc:.4f} ({class_correct[i]}/{class_total[i]})")

    if args.report_repr_metrics and all_pred_emb:
        pred_emb_all = torch.cat(all_pred_emb, dim=0)
        target_emb_all = torch.cat(all_target_emb, dim=0)
        pred_spectrum = covariance_spectrum(pred_emb_all)
        target_spectrum = covariance_spectrum(target_emb_all)
        pred_erank = effective_rank(pred_emb_all)
        target_erank = effective_rank(target_emb_all)
        var_ratio = variance_ratio(pred_emb_all, target_emb_all)
        topk = min(args.repr_topk_eigs, pred_spectrum.numel())
        print("\nRepresentation metrics:")
        print(f"  predictor effective rank: {pred_erank:.4f}")
        print(f"  target effective rank:    {target_erank:.4f}")
        print(f"  variance ratio (pred/target): {var_ratio:.4f}")
        print(f"  predictor covariance top-{topk} eigvals: {[round(v, 6) for v in pred_spectrum[:topk].tolist()]}")
        print(f"  target covariance top-{topk} eigvals:    {[round(v, 6) for v in target_spectrum[:topk].tolist()]}")


if __name__ == "__main__":
    main()
