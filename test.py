"""
Test/evaluate JEPA text classifier on Banking77.
Uses eb_jepa utilities: setup_device, load_checkpoint.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import Banking77Dataset, get_label_to_idx, get_labels, load_banking77
from eb_jepa_utils import load_checkpoint, setup_device
from model import JEPATextClassifier


def collate_fn(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache dir for Banking77")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    args = parser.parse_args()

    device = setup_device(args.device)
    ckpt_path = Path(args.checkpoint)

    # Data
    _, test_samples = load_banking77(cache_dir=args.cache_dir)
    label_to_idx = get_label_to_idx()
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    test_ds = Banking77Dataset(test_samples, label_to_idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = JEPATextClassifier(
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=device,
    )
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in test_loader:
            labels = labels.to(device)
            input_emb = model.encode_input(texts)
            pred_emb = model(input_emb)
            pred = (pred_emb @ model.head.label_embeddings.T).argmax(dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Test accuracy: {acc:.4f} ({correct}/{total})")

    # Per-class metrics
    n_classes = len(get_labels())
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


if __name__ == "__main__":
    main()
