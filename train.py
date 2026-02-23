"""
Train JEPA text classifier on Banking77.
Uses eb_jepa utilities: setup_device, setup_seed, save_checkpoint.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.banking77 import Banking77Dataset, get_labels, load_banking77_dataset
from utils.train import setup_device, setup_seed, save_checkpoint
from model import JEPATextClassifier, cosine_similarity_loss


def collate_fn(batch):
    """Collate (text, label_idx) into (list of texts, tensor of labels)."""
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels


def train_epoch(
    model: JEPATextClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="Train")
    for texts, labels in pbar:
        texts, labels = texts, labels.to(device)
        use_amp = device.type == "cuda"
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            input_emb = model.encode_input(texts)
            pred_emb = model(input_emb)
            loss = cosine_similarity_loss(pred_emb, model.head.label_embeddings, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        pbar.set_postfix(loss=loss.item())
    return total_loss / n if n > 0 else 0.0


@torch.no_grad()
def eval_epoch(
    model: JEPATextClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for texts, labels in tqdm(loader, desc="Eval"):
        labels = labels.to(device)
        input_emb = model.encode_input(texts)
        pred_emb = model(input_emb)
        loss = cosine_similarity_loss(pred_emb, model.head.label_embeddings, labels)
        total_loss += loss.item() * labels.size(0)
        pred = (pred_emb @ model.head.label_embeddings.T).argmax(dim=1)
        correct += (pred == labels).sum().item()
        n += labels.size(0)
    avg_loss = total_loss / n if n > 0 else 0.0
    acc = correct / n if n > 0 else 0.0
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache dir for Banking77 (default: ~/.cache/huggingface)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--predictor_hidden", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = setup_device(args.device)
    setup_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data
    ds_dict = load_banking77_dataset(cache_dir=args.cache_dir)
    labels = get_labels(cache_dir=args.cache_dir)
    train_ds = Banking77Dataset(ds_dict["train"])
    test_ds = Banking77Dataset(ds_dict["test"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Model
    model = JEPATextClassifier(
        labels=labels,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        predictor_hidden_dim=args.predictor_hidden,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)

    best_acc = 0.0
    for ep in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        eval_loss, eval_acc = eval_epoch(model, test_loader, device)
        print(f"Epoch {ep + 1}/{args.epochs} | train_loss={train_loss:.4f} | eval_loss={eval_loss:.4f} | eval_acc={eval_acc:.4f}")
        if eval_acc > best_acc:
            best_acc = eval_acc
            save_checkpoint(
                save_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=ep,
                eval_acc=eval_acc,
            )
            print(f"  Saved best model (acc={eval_acc:.4f})")

    print(f"Training done. Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
