"""
Test/evaluate JEPA text classifier on Banking77.
Uses eb_jepa utilities: setup_device, load_checkpoint.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.banking77 import Banking77Dataset, get_labels as get_banking77_labels, load_banking77_dataset
from dataset.clinc_oos import CLINCOOSDataset, get_labels as get_clinc_labels, load_clinc_oos_dataset
from utils.train import setup_device, load_checkpoint

from model import JEPATextClassifier

def collate_fn(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["banking77", "clinc_oos"], default="banking77")
    parser.add_argument("--clinc_config", type=str, choices=["plus", "small", "imbalanced"], default="plus")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache dir for Banking77")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--head_type", type=str, choices=["baseline", "moe"], default="baseline")
    parser.add_argument("--predictor_hidden", type=int, default=512)
    parser.add_argument("--baseline_dropout", type=float, default=0.0)
    parser.add_argument("--moe_num_experts", type=int, default=4)
    args = parser.parse_args()

    device = setup_device(args.device)
    ckpt_path = Path(args.checkpoint)

    # Data
    if args.dataset == "banking77":
        ds_dict = load_banking77_dataset(cache_dir=args.cache_dir)
        labels = get_banking77_labels(cache_dir=args.cache_dir)
        test_ds = Banking77Dataset(ds_dict["test"])
    else:
        ds_dict = load_clinc_oos_dataset(config=args.clinc_config, cache_dir=args.cache_dir)
        labels = get_clinc_labels(config=args.clinc_config, cache_dir=args.cache_dir)
        test_ds = CLINCOOSDataset(ds_dict["test"])
    idx_to_label = {i: lab for i, lab in enumerate(labels)}
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = JEPATextClassifier(
        labels=labels,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        head_type=args.head_type,
        predictor_hidden_dim=args.predictor_hidden,
        baseline_dropout=args.baseline_dropout,
        moe_num_experts=args.moe_num_experts,
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
