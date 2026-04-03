"""
VQA embedding plots: OpenCLIP encoder on ScienceQA and GQA.

Uses the same OpenCLIP (ViT-B-32) as run_vqa_models. Saves 4 separate PNGs per dataset:
  umap_vision_{dataset}_openclip.png         — vision (image) 2D, colored by correct choice
  umap_vision_overlay_{dataset}_openclip.png — vision 2D + correct-choice (text) overlay
  umap_fused_{dataset}_openclip.png         — fused (image+question) 2D, colored by correct choice
  umap_fused_overlay_{dataset}_openclip.png — fused 2D + correct-choice overlay

Output dir: runs/embedding_space_analysis/{dataset}/ (science_qa or gqa).

Example:
  python analysis/plot_vqa_embedding_umap.py --dataset science_qa --sample_size 2000
  python analysis/plot_vqa_embedding_umap.py --dataset gqa --sample_size 2000
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

# UMAP warns when random_state is set (n_jobs forced to 1); we want reproducible plots.
warnings.filterwarnings("ignore", message="n_jobs value .* overridden")
import torch

# Project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_jepa_baseline():
    """Load VisionQAJEPABaseline from jepa-baseline model (hyphen in path)."""
    import importlib.util
    path = _ROOT / "vision_qa" / "models" / "jepa-baseline" / "model.py"
    spec = importlib.util.spec_from_file_location("_vqa_plot_jepa", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "VisionQAJEPABaseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VQA 2D UMAP of OpenCLIP vision and fused (image+question) embeddings."
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
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--metric", type=str, default="cosine")
    p.add_argument("--cache_dir", type=str, default=None)
    return p.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


@torch.no_grad()
def load_and_encode_vqa(
    dataset: str,
    subset: str | None,
    split: str,
    sample_size: int,
    seed: int,
    batch_size: int,
    device: torch.device,
    clip_model: str,
    clip_pretrained: str,
    cache_dir: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load VQA dataset, sample, encode with OpenCLIP. Returns:
      image_emb [N, D], question_emb [N, D], fused [N, 2D],
      choice_embs [N, 5, D], answer_indices [N]
    """
    from vision_qa.data import load_vision_qa, make_collate_vision_qa
    from vision_qa.base import RunContext
    from torch.utils.data import DataLoader, Subset
    import random

    train_ds, val_ds, test_ds, _ = load_vision_qa(
        name=dataset,
        subset=subset,
        cache_dir=cache_dir,
        use_image=True,
    )
    split_ds = {"train": train_ds, "validation": val_ds, "test": test_ds}[split]

    n = len(split_ds)
    if n > sample_size:
        rng = random.Random(seed)
        indices = rng.sample(range(n), sample_size)
        split_ds = Subset(split_ds, indices)
    n = len(split_ds)

    # Build JEPA baseline only to use its OpenCLIP encoder
    VisionQAJEPABaseline = _load_jepa_baseline()
    model = VisionQAJEPABaseline(
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        hidden_dim=256,
        dropout=0.0,
    )
    model.to(device)
    model.eval()

    collate_fn = make_collate_vision_qa(image_transform=model.get_image_transform())
    loader = DataLoader(
        split_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    use_amp = device.type == "cuda"
    ctx = RunContext(device=device, use_amp=use_amp)

    all_img, all_q, all_fused = [], [], []
    all_choice = []
    all_ans = []

    for batch in loader:
        batch.images = batch.images.to(device)
        img_emb = model._encode_images(batch.images, ctx)
        q_emb = model._encode_texts(batch.questions, ctx)
        fused = torch.cat([img_emb, q_emb], dim=1)

        flat_choices = [c if c else " " for ch in batch.choices for c in ch]
        choice_emb = model._encode_texts(flat_choices, ctx)
        choice_emb = choice_emb / (choice_emb.norm(dim=-1, keepdim=True) + 1e-12)

        B = len(batch.choices)
        max_c_batch = max(len(ch) for ch in batch.choices)
        # Pad to global max 5 so tensors from all batches can be concatenated (ScienceQA has 2-5 choices)
        max_c = 5
        choice_embs = torch.zeros(B, max_c, model.embed_dim, device=device)
        offset = 0
        for i, ch in enumerate(batch.choices):
            nc = min(len(ch), max_c)
            choice_embs[i, :nc] = choice_emb[offset : offset + nc]
            offset += len(ch)

        all_img.append(img_emb.cpu().float())
        all_q.append(q_emb.cpu().float())
        all_fused.append(fused.cpu().float())
        all_choice.append(choice_embs.cpu().float())
        all_ans.append(batch.answer_indices)

    image_emb = torch.cat(all_img, dim=0)
    question_emb = torch.cat(all_q, dim=0)
    fused = torch.cat(all_fused, dim=0)
    choice_embs = torch.cat(all_choice, dim=0)
    answer_indices = torch.cat(all_ans, dim=0)

    return image_emb, question_emb, fused, choice_embs, answer_indices


def run_umap_2d(X: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as e:
        raise RuntimeError("pip install umap-learn") from e
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(X)


def run_umap_transform(fit_emb: np.ndarray, transform_emb: np.ndarray, n_neighbors: int, min_dist: float, metric: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import umap
    except ImportError as e:
        raise RuntimeError("pip install umap-learn") from e
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    fit_2d = reducer.fit_transform(fit_emb)
    trans_2d = reducer.transform(transform_emb)
    return fit_2d, trans_2d


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
    ax.set_title(f"Vision (image) embeddings — OpenCLIP — {dataset}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.colorbar(scatter, ax=ax, label="Correct choice (0–4)")
    fig.tight_layout()
    fig.savefig(out_dir / f"umap_vision_{dataset}_openclip.png")
    plt.close(fig)

    # 2. Vision 2D + correct choice (text) overlay
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    ax.scatter(image_2d[:, 0], image_2d[:, 1], s=5, alpha=0.2, c="lightgray", label="vision (images)")
    ax.scatter(target_2d_vision[:, 0], target_2d_vision[:, 1], s=80, alpha=0.9, c="darkred", marker="*", label="correct choice (text)", zorder=5)
    ax.set_title(f"Vision space + correct choice overlay — {dataset}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"umap_vision_overlay_{dataset}_openclip.png")
    plt.close(fig)

    # 3. Fused (image + question) embeddings, colored by correct choice
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    scatter = ax.scatter(fused_2d[:, 0], fused_2d[:, 1], c=ans, s=8, alpha=0.6, cmap="tab10")
    ax.set_title(f"Fused (image + question) embeddings — OpenCLIP — {dataset}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.colorbar(scatter, ax=ax, label="Correct choice (0–4)")
    fig.tight_layout()
    fig.savefig(out_dir / f"umap_fused_{dataset}_openclip.png")
    plt.close(fig)

    # 4. Fused 2D + correct choice (text) overlay
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    ax.scatter(fused_2d[:, 0], fused_2d[:, 1], s=5, alpha=0.2, c="lightgray", label="fused (image+q)")
    ax.scatter(target_2d_fused[:, 0], target_2d_fused[:, 1], s=80, alpha=0.9, c="darkred", marker="*", label="correct choice (text)", zorder=5)
    ax.set_title(f"Fused space + correct choice overlay — {dataset}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"umap_fused_overlay_{dataset}_openclip.png")
    plt.close(fig)

    print(f"Saved 4 PNGs to {out_dir}")


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
    # Correct choice embedding per sample [N, D]
    correct_choice_emb = torch.zeros(N, D)
    for i in range(N):
        idx = answer_indices[i].item()
        correct_choice_emb[i] = choice_embs[i, idx]

    print(f"Running UMAP (n_neighbors={args.n_neighbors})...")
    image_2d, target_2d_vision = run_umap_transform(
        image_emb.numpy(),
        correct_choice_emb.numpy(),
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        seed=args.seed,
    )
    # Fused is [N, 1024] (image+question); correct_choice is [N, 512]. Pad choice so transform sees same dim.
    fused_np = fused.numpy()
    choice_np = correct_choice_emb.numpy()
    choice_padded = np.pad(choice_np, ((0, 0), (0, fused_np.shape[1] - choice_np.shape[1])), mode="constant", constant_values=0)
    fused_2d, target_2d_fused = run_umap_transform(
        fused_np,
        choice_padded,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
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
