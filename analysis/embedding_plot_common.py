"""
Shared helpers for embedding-space 2D visualization (UMAP / t-SNE).
Loads data via text-classification data layer, builds encoder, returns input_emb, target_emb, labels, label_names.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F


def _ensure_paths() -> None:
    """Ensure project root and text-classification are on sys.path for imports."""
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    tc = root / "text-classification"
    if str(tc) not in sys.path:
        sys.path.insert(0, str(tc))


_ensure_paths()

from data import load_text_classification  # noqa: E402
from utils.encoders import OpenCLIPTextEncoder, SentenceTransformerEncoder  # noqa: E402


@dataclass
class EmbeddingPlotInputs:
    """Output of load_and_encode_for_plot: tensors and labels for 2D plotting."""

    input_emb: torch.Tensor   # [N, D] text embeddings
    target_emb: torch.Tensor  # [C, D] label embeddings
    labels: torch.Tensor      # [N] int label indices
    label_names: list[str]    # length C


def build_encoder_from_args(
    encoder: str,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "laion2b_s34b_b79k",
    sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> OpenCLIPTextEncoder | SentenceTransformerEncoder:
    """Build text encoder from string name and optional model args."""
    if encoder == "openclip":
        return OpenCLIPTextEncoder(name=clip_model, pretrained=clip_pretrained)
    if encoder == "minilm":
        return SentenceTransformerEncoder(model_name=sentence_model)
    raise ValueError(f"Unknown encoder: {encoder!r}. Use 'openclip' or 'minilm'.")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_texts_and_labels(
    dataset: str,
    subset: str | None,
    split: str,
    cache_dir: str | Path | None,
) -> tuple[list[str], list[int], list[str]]:
    """Load examples (texts, label indices) and label names using text-classification data layer."""
    train_ds, test_ds, get_labels = load_text_classification(
        name=dataset,
        subset=subset,
        cache_dir=cache_dir,
    )
    label_names = get_labels(train_ds.split)

    texts: list[str] = []
    labels_list: list[int] = []
    if split == "both":
        for i in range(len(train_ds)):
            t, l = train_ds[i]
            texts.append(t)
            labels_list.append(l)
        for i in range(len(test_ds)):
            t, l = test_ds[i]
            texts.append(t)
            labels_list.append(l)
    elif split == "train":
        for i in range(len(train_ds)):
            t, l = train_ds[i]
            texts.append(t)
            labels_list.append(l)
    else:
        for i in range(len(test_ds)):
            t, l = test_ds[i]
            texts.append(t)
            labels_list.append(l)

    return texts, labels_list, label_names


def stratified_sample(
    texts: list[str],
    labels: list[int],
    sample_size: int,
    max_per_class: int,
    seed: int,
) -> tuple[list[str], torch.Tensor]:
    """Return stratified sample of (texts, labels tensor)."""
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_label[y].append(i)
    for idxs in by_label.values():
        rng.shuffle(idxs)

    chosen: list[int] = []
    labels_sorted = sorted(by_label.keys())
    ptr = {k: 0 for k in labels_sorted}
    used: dict[int, int] = defaultdict(int)

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
def encode_texts(
    encoder: OpenCLIPTextEncoder | SentenceTransformerEncoder,
    texts: list[str],
    batch_size: int,
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """Encode list of texts to [N, D] tensor, normalized."""
    encoder.eval()
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            emb = encoder(batch)
        emb = emb.to(device=device, dtype=torch.float32)
        outs.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def build_label_embeddings(
    encoder: OpenCLIPTextEncoder | SentenceTransformerEncoder,
    label_names: list[str],
    device: torch.device,
    use_amp: bool = False,
) -> torch.Tensor:
    """Encode label names to [C, D], normalized (same as train's __build_label_embeddings__)."""
    encoder.eval()
    with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
        label_emb = encoder(label_names).to(device=device, dtype=torch.float32)
    label_emb = label_emb / (label_emb.norm(dim=-1, keepdim=True) + 1e-12)
    return label_emb.cpu()


def load_and_encode_for_plot(
    dataset: str,
    subset: str | None,
    split: str,
    cache_dir: str | Path | None,
    encoder: str,
    clip_model: str,
    clip_pretrained: str,
    sentence_model: str,
    sample_size: int,
    max_per_class: int,
    seed: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool = False,
) -> EmbeddingPlotInputs:
    """
    Load dataset, stratified sample, encode texts and label names with the given encoder.
    Returns EmbeddingPlotInputs with input_emb [N,D], target_emb [C,D], labels [N], label_names.
    """
    texts, labels_list, label_names = load_texts_and_labels(
        dataset=dataset,
        subset=subset,
        split=split,
        cache_dir=cache_dir,
    )
    sampled_texts, sampled_labels = stratified_sample(
        texts=texts,
        labels=labels_list,
        sample_size=sample_size,
        max_per_class=max_per_class,
        seed=seed,
    )

    enc = build_encoder_from_args(
        encoder=encoder,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        sentence_model=sentence_model,
    )
    enc = enc.to(device)

    input_emb = encode_texts(
        enc,
        sampled_texts,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )
    target_emb = build_label_embeddings(
        enc,
        label_names,
        device=device,
        use_amp=use_amp,
    )

    return EmbeddingPlotInputs(
        input_emb=input_emb,
        target_emb=target_emb,
        labels=sampled_labels,
        label_names=label_names,
    )
