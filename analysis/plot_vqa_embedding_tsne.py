"""
VQA embedding plots (t-SNE): OpenCLIP encoder on ScienceQA and GQA.

Same data and encoder as plot_vqa_embedding_umap.py, but uses t-SNE for 2D projection.
Saves 4 separate PNGs per dataset:
  tsne_vision_{dataset}_openclip.png         — vision (image) 2D, colored by correct choice
  tsne_vision_overlay_{dataset}_openclip.png — vision 2D + correct-choice (text) overlay
  tsne_fused_{dataset}_openclip.png          — fused (image+question) 2D, colored by correct choice
  tsne_fused_overlay_{dataset}_openclip.png  — fused 2D + correct-choice overlay

Output dir: runs/embedding_space_analysis/{dataset}/ (science_qa or gqa).

Example:
  python analysis/plot_vqa_embedding_tsne.py --dataset science_qa --sample_size 2000
  python analysis/plot_vqa_embedding_tsne.py --dataset gqa --sample_size 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse loader from UMAP script
from analysis.plot_vqa_embedding_umap import load_and_encode_vqa, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VQA 2D t-SNE of OpenCLIP vision and fused (image+question) embeddings."
    )
    p.add_argument("--dataset", choices=["science_qa", "gqa"], default="science_qa")
    p.add_argument("--subset", type=str, default=None, help="Dataset subset (GQA ignored; uses balanced)")
    p.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    p.add_argument("--sample_size", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, default="runs/embedding_space_analysis")
    p.add_argument("--clip_model", type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--max_iter", type=int, default=1000)
    return p.parse_args()


def run_tsne_transform(
    fit_emb: np.ndarray,
    transform_emb: np.ndarray,
    perplexity: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """t-SNE on stacked [fit_emb; transform_emb] so both lie in same 2D space. Returns (fit_2d, transform_2d)."""
    from sklearn.manifold import TSNE

    N = fit_emb.shape[0]
    stacked = np.vstack([fit_emb, transform_emb]).astype(np.float64)
    pp = min(perplexity, max(5, (2 * N - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=pp,
        random_state=seed,
        max_iter=max_iter,
        n_iter_without_progress=150,
    )
    out = tsne.fit_transform(stacked)
    return out[:N], out[N:]


def save_plots(
    out_dir: Path,
    dataset: str,
    image_2d: np.ndarray,
    target_2d_vision: np.ndarray,
    fused_2d: np.ndarray,
    target_2d_fused: np.ndarray,
    answer_indices: torch.Tensor,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ans = answer_indices.numpy()

    # 1. Vision (image) embeddings, colored by correct choice
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    scatter = ax.scatter(image_2d[:, 0], image_2d[:, 1], c=ans, s=8, alpha=0.6, cmap="tab10")
    ax.set_title(f"Vision (image) embeddings — OpenCLIP — {dataset} — t-SNE")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=ax, label="Correct choice (0–4)")
    fig.tight_layout()
    fig.savefig(out_dir / f"tsne_vision_{dataset}_openclip.png")
    plt.close(fig)

    # 2. Vision 2D + correct choice (text) overlay
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    ax.scatter(image_2d[:, 0], image_2d[:, 1], s=5, alpha=0.2, c="lightgray", label="vision (images)")
    ax.scatter(
        target_2d_vision[:, 0], target_2d_vision[:, 1],
        s=80, alpha=0.9, c="darkred", marker="*", label="correct choice (text)", zorder=5,
    )
    ax.set_title(f"Vision space + correct choice overlay — {dataset} — t-SNE")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"tsne_vision_overlay_{dataset}_openclip.png")
    plt.close(fig)

    # 3. Fused (image + question) embeddings, colored by correct choice
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    scatter = ax.scatter(fused_2d[:, 0], fused_2d[:, 1], c=ans, s=8, alpha=0.6, cmap="tab10")
    ax.set_title(f"Fused (image + question) embeddings — OpenCLIP — {dataset} — t-SNE")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=ax, label="Correct choice (0–4)")
    fig.tight_layout()
    fig.savefig(out_dir / f"tsne_fused_{dataset}_openclip.png")
    plt.close(fig)

    # 4. Fused 2D + correct choice (text) overlay
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    ax.scatter(fused_2d[:, 0], fused_2d[:, 1], s=5, alpha=0.2, c="lightgray", label="fused (image+q)")
    ax.scatter(
        target_2d_fused[:, 0], target_2d_fused[:, 1],
        s=80, alpha=0.9, c="darkred", marker="*", label="correct choice (text)", zorder=5,
    )
    ax.set_title(f"Fused space + correct choice overlay — {dataset} — t-SNE")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"tsne_fused_overlay_{dataset}_openclip.png")
    plt.close(fig)

    print(f"Saved 4 t-SNE PNGs to {out_dir}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    if args.dataset == "gqa":
        args.subset = "balanced"

    print(f"Loading {args.dataset} ({args.split}) and encoding with OpenCLIP {args.clip_model}...")
    image_emb, question_emb, fused, choice_embs, answer_indices = load_and_encode_vqa(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
        device=device,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        cache_dir=args.cache_dir,
    )

    N = image_emb.size(0)
    D = image_emb.size(1)
    correct_choice_emb = torch.zeros(N, D)
    for i in range(N):
        idx = answer_indices[i].item()
        correct_choice_emb[i] = choice_embs[i, idx]

    print(f"Running t-SNE (perplexity={args.perplexity}, max_iter={args.max_iter})...")
    image_2d, target_2d_vision = run_tsne_transform(
        image_emb.numpy(),
        correct_choice_emb.numpy(),
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    # Fused is [N, 1024] (image+question); correct_choice is [N, 512]. Pad choice to 1024 so we can stack.
    fused_np = fused.numpy()
    choice_np = correct_choice_emb.numpy()
    choice_padded = np.pad(choice_np, ((0, 0), (0, fused_np.shape[1] - choice_np.shape[1])), mode="constant", constant_values=0)
    fused_2d, target_2d_fused = run_tsne_transform(
        fused_np,
        choice_padded,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir) / args.dataset
    save_plots(
        out_dir=out_dir,
        dataset=args.dataset,
        image_2d=image_2d,
        target_2d_vision=target_2d_vision,
        fused_2d=fused_2d,
        target_2d_fused=target_2d_fused,
        answer_indices=answer_indices,
    )

    print("Done.")


if __name__ == "__main__":
    main()
