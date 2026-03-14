"""
2D UMAP embedding space analysis: input (text) embeddings and target (label) embeddings.

Given a dataset and encoder, builds a two-panel figure:
  (1) Input embeddings projected to 2D with UMAP, colored by class.
  (2) Same 2D space with target (label) embeddings overlaid via UMAP transform.

Example:
  python analysis/plot_embedding_umap.py --dataset banking77 --split test --encoder minilm --sample_size 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analysis.embedding_plot_common import (
    EmbeddingPlotInputs,
    load_and_encode_for_plot,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D UMAP visualization of input and target embeddings."
    )
    p.add_argument(
        "--dataset",
        choices=["banking77", "clinc_oos", "yahoo_answers"],
        default="banking77",
    )
    p.add_argument("--subset", type=str, default=None)
    p.add_argument("--split", choices=["train", "test", "both"], default="test")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--sample_size", type=int, default=2000)
    p.add_argument("--max_per_class", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--top_k_labels", type=int, default=12)
    p.add_argument("--encoder", choices=["openclip", "minilm"], default="minilm")
    p.add_argument("--clip_model", type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--sentence_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out_dir", type=str, default="runs/embedding_space_analysis")
    # UMAP-specific
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--metric", type=str, default="cosine")
    p.add_argument("--save_json", action="store_true", help="Save run params and summary to JSON")
    return p.parse_args()


def run_umap_2d(
    input_emb: torch.Tensor,
    target_emb: torch.Tensor,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit UMAP on input_emb, transform target_emb into same 2D space. Returns (input_2d, target_2d)."""
    try:
        import umap
    except ImportError as e:
        raise RuntimeError(
            "umap-learn is required for this script. Install with: pip install umap-learn"
        ) from e

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    X = input_emb.numpy()
    input_2d = reducer.fit_transform(X)
    target_2d = reducer.transform(target_emb.numpy())
    return input_2d, target_2d


def save_plots(
    out_dir: Path,
    data: EmbeddingPlotInputs,
    input_2d: np.ndarray,
    target_2d: np.ndarray,
    top_k_labels: int,
    dataset: str,
    encoder: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = data.labels
    label_names = data.label_names

    from collections import Counter
    label_counts = Counter(labels.tolist())
    top_labels = [k for k, _ in label_counts.most_common(top_k_labels)]
    top_set = set(top_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Panel 1: input embeddings in 2D, colored by class (top-k highlighted)
    ax = axes[0]
    other = np.array([int(y) not in top_set for y in labels.tolist()])
    ax.scatter(
        input_2d[other, 0], input_2d[other, 1],
        s=8, alpha=0.15, c="lightgray", label="other",
    )
    for lab in top_labels:
        mask = (labels == lab).numpy()
        ax.scatter(
            input_2d[mask, 0], input_2d[mask, 1],
            s=10, alpha=0.65, label=label_names[lab],
        )
    ax.set_title(f"Input embeddings (UMAP, top-{top_k_labels} labels)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Panel 2: same 2D space with target (label) points overlaid
    ax = axes[1]
    ax.scatter(
        input_2d[:, 0], input_2d[:, 1],
        s=5, alpha=0.2, c="lightgray", label="input",
    )
    ax.scatter(
        target_2d[:, 0], target_2d[:, 1],
        s=80, alpha=0.9, c="darkred", marker="*", label="target (labels)", zorder=5,
    )
    for i, name in enumerate(label_names):
        ax.annotate(
            name[:20] + ("..." if len(name) > 20 else ""),
            (target_2d[i, 0], target_2d[i, 1]),
            fontsize=6, alpha=0.8,
        )
    ax.set_title("Input + target (label) embeddings in same UMAP space")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"umap_embedding_space_{dataset}_{encoder}.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    data = load_and_encode_for_plot(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        cache_dir=args.cache_dir,
        encoder=args.encoder,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        sentence_model=args.sentence_model,
        sample_size=args.sample_size,
        max_per_class=args.max_per_class,
        seed=args.seed,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"Input embeddings: {data.input_emb.shape}, target: {data.target_emb.shape}")
    print(f"Running UMAP (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")

    input_2d, target_2d = run_umap_2d(
        data.input_emb,
        data.target_emb,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir) / args.dataset
    save_plots(
        out_dir=out_dir,
        data=data,
        input_2d=input_2d,
        target_2d=target_2d,
        top_k_labels=args.top_k_labels,
        dataset=args.dataset,
        encoder=args.encoder,
    )

    if args.save_json:
        metrics = {
            "dataset": args.dataset,
            "subset": args.subset,
            "split": args.split,
            "encoder": args.encoder,
            "sample_size": int(data.input_emb.size(0)),
            "n_classes": len(data.label_names),
            "umap": {
                "n_neighbors": args.n_neighbors,
                "min_dist": args.min_dist,
                "metric": args.metric,
            },
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"umap_embedding_metrics_{args.dataset}_{args.encoder}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote {json_path}")

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
