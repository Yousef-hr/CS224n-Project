"""
2D t-SNE embedding space analysis: input (text) embeddings and target (label) embeddings.

Given a dataset and encoder, builds a two-panel figure:
  (1) Input embeddings projected to 2D with t-SNE, colored by class.
  (2) Target (label) embeddings projected to 2D with t-SNE (separate fit; no out-of-sample transform).

Example:
  python analysis/plot_embedding_tsne.py --dataset banking77 --split test --encoder minilm --sample_size 2000
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
        description="2D t-SNE visualization of input and target embeddings."
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
    # t-SNE-specific
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--learning_rate", type=float, default=200.0)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--save_json", action="store_true", help="Save run params and summary to JSON")
    return p.parse_args()


def run_tsne_2d(
    input_emb: torch.Tensor,
    target_emb: torch.Tensor,
    perplexity: float,
    learning_rate: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run t-SNE on input_emb and separately on target_emb (no shared space). Returns (input_2d, target_2d)."""
    try:
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise RuntimeError(
            "scikit-learn is required for t-SNE. Install with: pip install scikit-learn"
        ) from e

    X = input_emb.numpy()
    Y = target_emb.numpy()

    # Perplexity must be < n_samples. For small target set, use min(perplexity, n_classes-1)
    perplexity_input = min(perplexity, max(2, X.shape[0] - 1) // 3)
    perplexity_target = min(perplexity, max(2, Y.shape[0] - 1) // 3)

    tsne_input = TSNE(
        n_components=2,
        perplexity=perplexity_input,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=seed,
        init="pca",
    )
    input_2d = tsne_input.fit_transform(X)

    tsne_target = TSNE(
        n_components=2,
        perplexity=perplexity_target,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=seed + 1,
        init="pca",
    )
    target_2d = tsne_target.fit_transform(Y)

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
    ax.set_title(f"Input embeddings (t-SNE, top-{top_k_labels} labels)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Panel 2: target (label) embeddings in their own 2D space
    ax = axes[1]
    ax.scatter(target_2d[:, 0], target_2d[:, 1], s=80, alpha=0.8, c="darkred", marker="*")
    for i, name in enumerate(label_names):
        ax.annotate(
            name[:20] + ("..." if len(name) > 20 else ""),
            (target_2d[i, 0], target_2d[i, 1]),
            fontsize=6, alpha=0.8,
        )
    ax.set_title("Target (label) embeddings (t-SNE, separate space)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    fig.tight_layout()
    fig.savefig(out_dir / f"tsne_embedding_space_{dataset}_{encoder}.png")
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
    print(f"Running t-SNE (perplexity={args.perplexity}, max_iter={args.max_iter})...")

    input_2d, target_2d = run_tsne_2d(
        data.input_emb,
        data.target_emb,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
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
            "tsne": {
                "perplexity": args.perplexity,
                "learning_rate": args.learning_rate,
                "max_iter": args.max_iter,
            },
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"tsne_embedding_metrics_{args.dataset}_{args.encoder}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote {json_path}")

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
