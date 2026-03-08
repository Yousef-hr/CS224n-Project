"""
Composable CLI for text classification train/eval scripts.
Add shared arg groups then model-specific args; build specs from parsed args.
"""

import argparse
from pathlib import Path

from .base import DataSpec, EvalSpec, TrainSpec


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=str, choices=["yahoo_answers", "banking77", "clinc_oos"], default="yahoo_answers")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config (e.g. CLINC-OOS: plus, small, imbalanced)")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--embedding_cache_dir", type=str, default=None, help="Directory to store/reuse frozen text embeddings")
    parser.add_argument("--precompute_batch_size", type=int, default=512)


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--metrics_csv", type=str, default=None)


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no_strict", action="store_true", help="Load checkpoint with strict=False")
    parser.add_argument("--report_repr_metrics", action="store_true")
    parser.add_argument("--repr_topk_eigs", type=int, default=10)


def add_encoder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder", type=str, choices=["openclip", "minilm"], default="openclip", help="Text encoder: openclip (CLIP) or minilm (SentenceTransformer)")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32", help="OpenCLIP model name (used when --encoder openclip)")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k", help="OpenCLIP pretrained (used when --encoder openclip)")
    parser.add_argument("--sentence_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-transformers model (used when --encoder minilm)")


def build_data_spec(args: argparse.Namespace) -> DataSpec:
    subset = getattr(args, "subset", None) or getattr(args, "clinc_config", None)
    if args.dataset == "clinc_oos" and subset is None:
        subset = "plus"
    
    return DataSpec(
        dataset=args.dataset,
        subset=subset,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=getattr(args, "num_workers", 0),
        embedding_cache_dir=getattr(args, "embedding_cache_dir", None),
        precompute_batch_size=getattr(args, "precompute_batch_size", 512),
        encoder=getattr(args, "encoder", "openclip"),
        clip_model=getattr(args, "clip_model", "ViT-B-32"),
        clip_pretrained=getattr(args, "clip_pretrained", "laion2b_s34b_b79k"),
        sentence_model=getattr(args, "sentence_model", "sentence-transformers/all-MiniLM-L6-v2"),
    )


def build_train_spec(args: argparse.Namespace, metrics_csv_default: str | Path | None = None) -> TrainSpec:
    save_dir = getattr(args, "save_dir", "checkpoints")
    metrics_csv = getattr(args, "metrics_csv", None) or metrics_csv_default or (Path(save_dir) / "training_metrics.csv")
    
    return TrainSpec(
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.0),
        save_dir=save_dir,
        metrics_csv=metrics_csv,
    )


def build_eval_spec(args: argparse.Namespace) -> EvalSpec:
    return EvalSpec(
        device=args.device,
        report_repr_metrics=getattr(args, "report_repr_metrics", False),
        repr_topk_eigs=getattr(args, "repr_topk_eigs", 10),
    )


def make_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_train_args(parser)
    add_encoder_args(parser)
    return parser


def make_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_eval_args(parser)
    add_encoder_args(parser)
    return parser
