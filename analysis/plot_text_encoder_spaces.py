"""
Compare text embedding spaces for MiniLM vs OpenCLIP.

Outputs:
  - Side-by-side 2D PCA scatter plot
  - Pairwise cosine similarity histogram
  - Covariance spectrum plot
  - Summary metrics JSON

Example:
  python analysis/plot_text_encoder_spaces.py --dataset banking77 --split test --sample_size 2000
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from utils.encoders import OpenCLIPTextEncoder, SentenceTransformerEncoder
from utils.metrics import covariance_spectrum, effective_rank, linear_cka


DATASET_RECIPES = {
    "banking77": {
        "id": "legacy-datasets/banking77",
        "default_subset": None,
        "label_key": "label",
        "text_key": "text",
    },
    "clinc_oos": {
        "id": "clinc_oos",
        "default_subset": "plus",
        "label_key": "intent",
        "text_key": "text",
    },
    "yahoo_answers": {
        "id": "yahoo_answers_topics",
        "default_subset": None,
        "label_key": "topic",
        "text_key": "question_title",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot and compare MiniLM vs OpenCLIP embedding spaces.")
    p.add_argument("--dataset", choices=list(DATASET_RECIPES.keys()), default="banking77")
    p.add_argument("--subset", type=str, default=None)
    p.add_argument("--split", choices=["train", "test", "both"], default="test")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--sample_size", type=int, default=2000)
    p.add_argument("--max_per_class", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--top_k_labels", type=int, default=12)
    p.add_argument("--clip_model", type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--sentence_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out_dir", type=str, default="runs/embedding_space_analysis")
    return p.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(raw)


def load_text_examples(
    dataset: str,
    subset: str | None,
    split: str,
    cache_dir: str | None,
) -> tuple[list[str], list[int], list[str]]:
    recipe = DATASET_RECIPES[dataset]
    ds_id = recipe["id"]
    sub = subset if subset is not None else recipe["default_subset"]
    label_key = recipe["label_key"]
    text_key = recipe["text_key"]

    if sub is None:
        ds = load_dataset(ds_id, cache_dir=cache_dir)
    else:
        ds = load_dataset(ds_id, sub, cache_dir=cache_dir)

    splits = ["train", "test"] if split == "both" else [split]
    texts: list[str] = []
    labels: list[int] = []
    for s in splits:
        for ex in ds[s]:
            text = ex[text_key]
            if dataset == "yahoo_answers":
                # Use title + body when available for a richer sentence embedding.
                body = ex.get("question_content", "")
                if isinstance(body, str) and body.strip():
                    text = f"{text}. {body}"
            texts.append(text)
            labels.append(int(ex[label_key]))

    label_names = list(ds["train"].features[label_key].names)  # type: ignore[attr-defined]
    return texts, labels, label_names


def stratified_sample(
    texts: list[str],
    labels: list[int],
    sample_size: int,
    max_per_class: int,
    seed: int,
) -> tuple[list[str], torch.Tensor]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_label[y].append(i)
    for idxs in by_label.values():
        rng.shuffle(idxs)

    chosen: list[int] = []
    labels_sorted = sorted(by_label.keys())
    ptr = {k: 0 for k in labels_sorted}
    used = Counter()

    while len(chosen) < min(sample_size, len(texts)):
        progressed = False
        for k in labels_sorted:
            if len(chosen) >= sample_size:
                break
            if used[k] >= max_per_class:
                continue
            p = ptr[k]
            idxs = by_label[k]
            if p >= len(idxs):
                continue
            chosen.append(idxs[p])
            ptr[k] += 1
            used[k] += 1
            progressed = True
        if not progressed:
            break

    sampled_texts = [texts[i] for i in chosen]
    sampled_labels = torch.tensor([labels[i] for i in chosen], dtype=torch.long)
    return sampled_texts, sampled_labels


@torch.no_grad()
def encode_openclip(
    texts: list[str],
    batch_size: int,
    device: torch.device,
    clip_model: str,
    clip_pretrained: str,
) -> torch.Tensor:
    enc = OpenCLIPTextEncoder(name=clip_model, pretrained=clip_pretrained).to(device)
    enc.eval()
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = enc.tokenizer(batch).to(device)
        emb = enc.model.encode_text(tokens).float()
        outs.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def encode_minilm(
    texts: list[str],
    batch_size: int,
    device: torch.device,
    sentence_model: str,
) -> torch.Tensor:
    enc = SentenceTransformerEncoder(model_name=sentence_model).to(device)
    enc.eval()
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        toks = enc.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = enc.model(**toks)
        token_emb = out[0]
        mask = toks["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
        pooled = torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        outs.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(outs, dim=0)


def pca_2d(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    q = min(2, min(x.shape) - 1)
    if q < 1:
        return torch.zeros(x.size(0), 2)
    u, s, _ = torch.pca_lowrank(x, q=2)
    z = u[:, :2] * s[:2]
    if z.size(1) == 1:
        z = torch.cat([z, torch.zeros_like(z)], dim=1)
    return z


def centroid_metrics(emb: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    n_classes = int(labels.max().item()) + 1
    d = emb.size(1)
    centroids = torch.zeros(n_classes, d)
    counts = torch.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            centroids[c] = emb[mask].mean(dim=0)
            counts[c] = int(mask.sum().item())
    centroids = F.normalize(centroids, dim=-1)

    own = torch.sum(emb * centroids[labels], dim=1).mean().item()
    valid = counts > 0
    c = centroids[valid]
    sim = c @ c.T
    if sim.numel() <= 1:
        inter = 0.0
    else:
        inter = (sim.sum() - torch.diag(sim).sum()).item() / max(c.size(0) * (c.size(0) - 1), 1)
    return {
        "mean_cosine_to_own_centroid": float(own),
        "mean_cosine_between_class_centroids": float(inter),
    }


def save_plots(
    out_dir: Path,
    label_names: list[str],
    labels: torch.Tensor,
    openclip_emb: torch.Tensor,
    minilm_emb: torch.Tensor,
    top_k_labels: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    label_counts = Counter(labels.tolist())
    top_labels = [k for k, _ in label_counts.most_common(top_k_labels)]
    top_set = set(top_labels)
    proj_openclip = pca_2d(openclip_emb)
    proj_minilm = pca_2d(minilm_emb)

    # 1) PCA scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    for ax, z, title in [
        (axes[0], proj_openclip, "OpenCLIP"),
        (axes[1], proj_minilm, "MiniLM"),
    ]:
        other = ~torch.tensor([int(y) in top_set for y in labels.tolist()], dtype=torch.bool)
        ax.scatter(z[other, 0], z[other, 1], s=8, alpha=0.15, c="lightgray", label="other")
        for lab in top_labels:
            mask = labels == lab
            ax.scatter(z[mask, 0], z[mask, 1], s=10, alpha=0.65, label=label_names[lab])
        ax.set_title(f"{title} embedding PCA (top-{top_k_labels} labels highlighted)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    handles, labels_text = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_dir / "pca_scatter_openclip_vs_minilm.png")
    plt.close(fig)

    # 2) Pairwise cosine histogram
    rng = torch.Generator().manual_seed(0)
    n = openclip_emb.size(0)
    pairs = min(20000, n * (n - 1) // 2)
    i_idx = torch.randint(0, n, (pairs,), generator=rng)
    j_idx = torch.randint(0, n, (pairs,), generator=rng)
    valid = i_idx != j_idx
    i_idx, j_idx = i_idx[valid], j_idx[valid]
    cos_open = torch.sum(openclip_emb[i_idx] * openclip_emb[j_idx], dim=1).numpy()
    cos_mini = torch.sum(minilm_emb[i_idx] * minilm_emb[j_idx], dim=1).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
    ax.hist(cos_open, bins=80, alpha=0.5, density=True, label="OpenCLIP")
    ax.hist(cos_mini, bins=80, alpha=0.5, density=True, label="MiniLM")
    ax.set_title("Pairwise cosine similarity distribution")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pairwise_cosine_hist_openclip_vs_minilm.png")
    plt.close(fig)

    # 3) Covariance spectrum
    eig_open = covariance_spectrum(openclip_emb).cpu().numpy()
    eig_mini = covariance_spectrum(minilm_emb).cpu().numpy()
    topk = min(128, len(eig_open), len(eig_mini))
    x = np.arange(1, topk + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
    ax.plot(x, eig_open[:topk] / max(eig_open[0], 1e-12), label="OpenCLIP")
    ax.plot(x, eig_mini[:topk] / max(eig_mini[0], 1e-12), label="MiniLM")
    ax.set_title("Normalized covariance eigenvalue spectrum")
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eig / eig_1")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cov_spectrum_openclip_vs_minilm.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    texts, labels, label_names = load_text_examples(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    sampled_texts, sampled_labels = stratified_sample(
        texts=texts,
        labels=labels,
        sample_size=args.sample_size,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    print(f"Loaded {len(texts)} examples, sampled {len(sampled_texts)} for analysis.")
    print(f"Encoding on device: {device}")

    openclip_emb = encode_openclip(
        sampled_texts,
        batch_size=args.batch_size,
        device=device,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
    minilm_emb = encode_minilm(
        sampled_texts,
        batch_size=args.batch_size,
        device=device,
        sentence_model=args.sentence_model,
    )

    # Safety normalize (both encoders should already be normalized).
    openclip_emb = F.normalize(openclip_emb, dim=-1)
    minilm_emb = F.normalize(minilm_emb, dim=-1)

    out_dir = Path(args.out_dir) / args.dataset
    save_plots(
        out_dir=out_dir,
        label_names=label_names,
        labels=sampled_labels,
        openclip_emb=openclip_emb,
        minilm_emb=minilm_emb,
        top_k_labels=args.top_k_labels,
    )

    metrics = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "sample_size": int(len(sampled_texts)),
        "openclip": {
            "effective_rank": effective_rank(openclip_emb),
            "top1_cov_eig": float(covariance_spectrum(openclip_emb)[0].item()),
            **centroid_metrics(openclip_emb, sampled_labels),
        },
        "minilm": {
            "effective_rank": effective_rank(minilm_emb),
            "top1_cov_eig": float(covariance_spectrum(minilm_emb)[0].item()),
            **centroid_metrics(minilm_emb, sampled_labels),
        },
        "cross_encoder_alignment": {
            "linear_cka": float(linear_cka(openclip_emb, minilm_emb)),
            "mean_cosine_between_encoders_same_text": float(
                torch.sum(openclip_emb * minilm_emb, dim=1).mean().item()
            ),
        },
    }

    metrics_path = out_dir / "embedding_space_metrics.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote plots and metrics to: {out_dir}")
    print(f"Metrics JSON: {metrics_path}")
    print(
        "Tip: compare effective_rank, centroid separation, and cosine hist spread "
        "against JEPA/SOTA behavior."
    )


if __name__ == "__main__":
    main()
