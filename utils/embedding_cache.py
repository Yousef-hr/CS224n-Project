from __future__ import annotations

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


def _encode_dataset_split(split_ds, encoder, device: torch.device, batch_size: int, desc: str):
    loader = DataLoader(split_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_text, num_workers=0)
    all_embeddings = []
    all_labels = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for texts, labels in tqdm(loader, desc=desc):
            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                emb = encoder(texts)
            all_embeddings.append(emb.detach().cpu().to(torch.float16))
            all_labels.append(labels.cpu())
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def _cache_path(cache_dir: Path, dataset_id: str, encoder_id: str) -> Path:
    filename = f"{_sanitize(dataset_id)}__{_sanitize(encoder_id)}.pt"
    return cache_dir / filename


def get_or_build_text_embedding_cache(
    *,
    cache_dir: str | Path,
    dataset_id: str,
    encoder_id: str,
    train_ds,
    test_ds,
    labels_list: list[str],
    encoder,
    device: torch.device,
    precompute_batch_size: int = 512,
) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, dataset_id, encoder_id)
    if path.exists():
        print(f"Loading precomputed text embeddings: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    print(f"Building precomputed text embeddings: {path}")
    train_emb, train_labels = _encode_dataset_split(train_ds, encoder, device, precompute_batch_size, "Precompute train")
    test_emb, test_labels = _encode_dataset_split(test_ds, encoder, device, precompute_batch_size, "Precompute test")
    with torch.no_grad():
        label_emb = encoder(labels_list).detach().cpu().to(torch.float16)

    payload = {
        "dataset_id": dataset_id,
        "encoder_id": encoder_id,
        "train_embeddings": train_emb,
        "train_labels": train_labels,
        "test_embeddings": test_emb,
        "test_labels": test_labels,
        "label_embeddings": label_emb,
    }
    torch.save(payload, path)
    print(f"Saved precomputed text embeddings: {path}")
    return payload


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
