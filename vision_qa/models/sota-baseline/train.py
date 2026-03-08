"""
Train vision QA baseline (frozen CLIP image+text -> MLP -> logits).
"""

import argparse
from pathlib import Path

from vision_qa.base import DataSpec, TrainSpec
from vision_qa.train import run_vision_qa_train
from vision_qa.data import make_collate_vision_qa

from .model import VisionQABaseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="science_qa", choices=["science_qa"])
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--metrics_csv", type=str, default=None)
    parser.add_argument("--use_image", action="store_true", default=True, help="Use image modality (default True)")
    parser.add_argument("--no_use_image", action="store_false", dest="use_image", help="Text-only")

    args = parser.parse_args()

    model = VisionQABaseline(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
    )

    image_transform = model.get_image_transform()

    data = DataSpec(
        dataset=args.dataset,
        subset=args.subset,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=0,
        use_image=args.use_image,
    )

    train = TrainSpec(
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        metrics_csv=args.metrics_csv or str(Path(args.save_dir) / "training_metrics_vision_qa_baseline.csv"),
    )

    run_vision_qa_train(
        model=model,
        data=data,
        train=train,
        image_transform=image_transform,
        save_name="best_vision_qa_baseline.pt",
    )


if __name__ == "__main__":
    main()
