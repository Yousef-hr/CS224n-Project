"""Evaluate Deep JEPA VQA (load checkpoint, run on test/val)."""

import argparse
from pathlib import Path

from vision_qa.base import DataSpec, EvalSpec
from vision_qa.eval import run_vision_qa_eval

from .model import VisionQAJEPADeep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="science_qa", choices=["science_qa"])
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--report_breakdown", action="store_true")
    parser.add_argument("--no_strict", action="store_true")
    parser.add_argument("--use_image", action="store_true", default=True)
    parser.add_argument("--no_use_image", action="store_false", dest="use_image")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args()

    model = VisionQAJEPADeep(
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    )

    data = DataSpec(
        dataset=args.dataset,
        subset=args.subset,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=0,
        use_image=args.use_image,
    )
    eval_spec = EvalSpec(device=args.device, report_breakdown=args.report_breakdown)

    run_vision_qa_eval(
        model=model,
        data=data,
        eval_spec=eval_spec,
        image_transform=model.get_image_transform(),
        checkpoint=Path(args.checkpoint),
        strict=not args.no_strict,
        split=args.split,
    )


if __name__ == "__main__":
    main()
