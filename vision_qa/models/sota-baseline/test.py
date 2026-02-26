"""
Evaluate vision QA baseline (load checkpoint, run on test/val).
"""

import argparse
from pathlib import Path

from vision_qa.base import DataSpec, EvalSpec
from vision_qa.eval import run_vision_qa_eval

from .model import VisionQABaseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="science_qa", choices=["science_qa"])
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default=None, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--report_breakdown", action="store_true", help="Report accuracy by subject/grade")
    parser.add_argument("--no_strict", action="store_true", help="Load checkpoint with strict=False")
    parser.add_argument("--use_image", action="store_true", default=True)
    parser.add_argument("--no_use_image", action="store_false", dest="use_image")

    args = parser.parse_args()

    model = VisionQABaseline(
        clip_model="ViT-B-32",
        clip_pretrained="laion2b_s34b_b79k",
        hidden_dim=512,
        dropout=0.1,
        lr=3e-4,
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

    eval_spec = EvalSpec(
        device=args.device,
        report_breakdown=args.report_breakdown,
    )

    run_vision_qa_eval(
        model=model,
        data=data,
        eval_spec=eval_spec,
        image_transform=image_transform,
        checkpoint=Path(args.checkpoint),
        strict=not args.no_strict,
        split=args.split,
    )


if __name__ == "__main__":
    main()
