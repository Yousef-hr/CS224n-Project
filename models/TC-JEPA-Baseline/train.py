"""
Train JEPA baseline text classifier (frozen encoder + MLP predictor).
Loss: 1 - cosine_similarity(pred_emb, label_emb). Optional SigReg.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project root and model dir on path for imports
_root = Path(__file__).resolve().parent.parent.parent
_model_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_model_dir))

from dataset.banking77 import Banking77Dataset, get_labels as get_banking77_labels, load_banking77_dataset
from dataset.clinc_oos import CLINCOOSDataset, get_labels as get_clinc_labels, load_clinc_oos_dataset
from encoders.OpenCLIP import OpenCLIPTextEncoder
from utils.train import setup_device, setup_seed, save_checkpoint
from utils.losses import cosine_similarity_loss

from model import BaselineJEPATextClassifier

def collate_fn(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels


def train_epoch(model, loader, optimizer, device, label_embeddings):
    model.train()
    total_loss = 0.0
    n = 0

    for texts, labels in tqdm(loader, desc="Train"):
        labels = labels.to(device)
        use_amp = device.type == "cuda"

        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            input_emb = model.encode_text(texts)
            pred_emb = model(input_emb)
            pred_emb = pred_emb / pred_emb.norm(dim=-1, keepdim=True)
            targets = label_embeddings[labels]

            loss = cosine_similarity_loss(pred_emb, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

    return total_loss / n if n else 0.0


@torch.no_grad()
def eval_epoch(model, loader, device, label_embeddings):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    for texts, labels in tqdm(loader, desc="Eval"):
        labels = labels.to(device)

        input_emb = model.encode_text(texts)
        pred_emb = model(input_emb)
        pred_emb = pred_emb / pred_emb.norm(dim=-1, keepdim=True)
        targets = label_embeddings[labels]

        total_loss += cosine_similarity_loss(pred_emb, targets).item() * labels.size(0)
        logits = pred_emb @ label_embeddings.T
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += labels.size(0)

    return total_loss / n if n else 0.0, correct / n if n else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["banking77", "clinc_oos"], default="banking77")
    parser.add_argument("--clinc_config", type=str, choices=["plus", "small", "imbalanced"], default="plus")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--predictor_hidden", type=int, default=512)
    parser.add_argument("--baseline_dropout", type=float, default=0.0)
    parser.add_argument("--use_sigreg", action="store_true")
    parser.add_argument("--sigreg_weight", type=float, default=0.1)
    parser.add_argument("--sigreg_target_std", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = setup_device(args.device)
    setup_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "banking77":
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

    model = BaselineJEPATextClassifier(
        encoder=encoder,
        predictor_hidden_dim=args.predictor_hidden,
        baseline_dropout=args.baseline_dropout,
        device=device,
    ).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)

    best_acc = 0.0
    for ep in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device, label_embeddings)
        eval_loss, eval_acc = eval_epoch(model, test_loader, device, label_embeddings)
        print(f"Epoch {ep + 1}/{args.epochs} | train_loss={loss:.4f} | eval_loss={eval_loss:.4f} | eval_acc={eval_acc:.4f}")
        if eval_acc > best_acc:
            best_acc = eval_acc
            save_checkpoint(save_dir / "best_jepa_baseline.pt", model=model, optimizer=optimizer, epoch=ep, eval_acc=eval_acc)
            print(f"  Saved best (acc={eval_acc:.4f})")
    print(f"Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
