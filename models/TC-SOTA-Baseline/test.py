"""
Test/evaluate SOTA baseline text classifier (frozen encoder + MLP → logits) on Banking77 / CLINC-OOS.
Uses cross-entropy classifier; reports accuracy and per-class accuracy.
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

from model import SOTABaselineTextClassifier


def collate_fn(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["yahoo_answers", "banking77", "clinc_oos"], default="yahoo_answers")
    parser.add_argument("--clinc_config", type=str, choices=["plus", "small", "imbalanced"], default="plus")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache dir")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_sota_baseline.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[1024], help="MLP hidden dims (must match training)")
    parser.add_argument("--dropout", type=float, default=0.1)
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
    num_classes = len(labels_list)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Encoder
    encoder = OpenCLIPTextEncoder(
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=device,
    )

    # Model (logits head, no label embeddings)
    model = SOTABaselineTextClassifier(
        encoder=encoder,
        num_classes=num_classes,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        device=device,
    ).to(device)
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
            input_emb = model.encode_text(texts)
            logits = model(input_emb)
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            correct += (pred == labels).sum().item()
            total += labels.size(0)

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


if __name__ == "__main__":
    main()
