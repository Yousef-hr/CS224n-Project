from __future__ import annotations

import gc
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name)


def _collate_text(batch):
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels


def _get_embed_dim(encoder, device: torch.device) -> int:
    dim = getattr(encoder, "embed_dim", None)
    if dim is not None:
        return dim
    with torch.no_grad():
        probe = encoder(["hello"])
    return probe.shape[-1]


def _encode_dataset_split(split_ds, encoder, device: torch.device, batch_size: int, desc: str):
    """Encode a split into a pre-allocated tensor (no list + cat memory spike)."""
    n = len(split_ds)
    embed_dim = _get_embed_dim(encoder, device)
    all_embeddings = torch.empty(n, embed_dim, dtype=torch.float16)
    all_labels = torch.empty(n, dtype=torch.long)

    loader = DataLoader(split_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_text, num_workers=0)
    use_amp = device.type == "cuda"
    offset = 0
    with torch.no_grad():
        for texts, labels in tqdm(loader, desc=desc):
            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                emb = encoder(texts)
            bs = emb.size(0)
            all_embeddings[offset:offset + bs] = emb.detach().cpu().to(torch.float16)
            all_labels[offset:offset + bs] = labels.cpu()
            offset += bs
    return all_embeddings[:offset], all_labels[:offset]


def _cache_prefix(cache_dir: Path, dataset_id: str, clip_model: str, clip_pretrained: str) -> str:
    return f"{_sanitize(dataset_id)}__{_sanitize(clip_model)}__{_sanitize(clip_pretrained)}"


def _split_paths(cache_dir: Path, prefix: str):
    return (
        cache_dir / f"{prefix}__train.pt",
        cache_dir / f"{prefix}__test.pt",
        cache_dir / f"{prefix}__labels.pt",
    )


def get_or_build_text_embedding_cache(
    *,
    cache_dir: str | Path,
    dataset_id: str,
    clip_model: str,
    clip_pretrained: str,
    train_ds,
    test_ds,
    labels_list: list[str],
    encoder,
    device: torch.device,
    precompute_batch_size: int = 512,
) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = _cache_prefix(cache_dir, dataset_id, clip_model, clip_pretrained)
    train_path, test_path, label_path = _split_paths(cache_dir, prefix)

    if train_path.exists() and test_path.exists() and label_path.exists():
        print(f"Loading precomputed text embeddings from {cache_dir}/{prefix}__*.pt")
        train_data = torch.load(train_path, map_location="cpu", weights_only=False)
        test_data = torch.load(test_path, map_location="cpu", weights_only=False)
        return {
            "dataset_id": dataset_id,
            "clip_model": clip_model,
            "clip_pretrained": clip_pretrained,
            "train_embeddings": train_data["emb"],
            "train_labels": train_data["labels"],
            "test_embeddings": test_data["emb"],
            "test_labels": test_data["labels"],
            "label_embeddings": torch.load(label_path, map_location="cpu", weights_only=False),
        }

    print(f"Building precomputed text embeddings: {cache_dir}/{prefix}__*.pt")

    with torch.no_grad():
        label_emb = encoder(labels_list).detach().cpu().to(torch.float16)
    torch.save(label_emb, label_path)
    print(f"  Saved label embeddings: {label_path}")

    train_emb, train_labels = _encode_dataset_split(train_ds, encoder, device, precompute_batch_size, "Precompute train")
    torch.save({"emb": train_emb, "labels": train_labels}, train_path)
    print(f"  Saved train embeddings: {train_path}  ({train_emb.shape})")
    del train_emb, train_labels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    test_emb, test_labels = _encode_dataset_split(test_ds, encoder, device, precompute_batch_size, "Precompute test")
    torch.save({"emb": test_emb, "labels": test_labels}, test_path)
    print(f"  Saved test embeddings: {test_path}  ({test_emb.shape})")
    del test_emb, test_labels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("  Loading saved splits back...")
    train_data = torch.load(train_path, map_location="cpu", weights_only=False)
    test_data = torch.load(test_path, map_location="cpu", weights_only=False)

    return {
        "dataset_id": dataset_id,
        "clip_model": clip_model,
        "clip_pretrained": clip_pretrained,
        "train_embeddings": train_data["emb"],
        "train_labels": train_data["labels"],
        "test_embeddings": test_data["emb"],
        "test_labels": test_data["labels"],
        "label_embeddings": label_emb,
    }


def build_cached_loaders(cache_payload: dict, batch_size: int):
    train_loader = DataLoader(
        TensorDataset(cache_payload["train_embeddings"], cache_payload["train_labels"]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(cache_payload["test_embeddings"], cache_payload["test_labels"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader
