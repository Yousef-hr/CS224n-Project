"""
Precompute and cache frozen CLIP embeddings for Vision QA datasets.

Saves image, question, and per-choice CLIP embeddings to disk so that
subsequent training runs skip all encoder forward passes entirely.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vision_qa.base import RunContext
from vision_qa.data import make_collate_vision_qa

MAX_CHOICES = 5


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", name)


def _split_path(cache_dir: Path, prefix: str, split: str) -> Path:
    return cache_dir / f"{prefix}__{split}.pt"


def _cache_prefix(dataset_id: str, clip_model: str, clip_pretrained: str) -> str:
    return f"{_sanitize(dataset_id)}__{_sanitize(clip_model)}__{_sanitize(clip_pretrained)}"


def vqa_cache_exists(
    cache_dir: str | Path,
    dataset_id: str,
    clip_model: str,
    clip_pretrained: str,
) -> bool:
    cache_dir = Path(cache_dir)
    prefix = _cache_prefix(dataset_id, clip_model, clip_pretrained)
    return all(
        _split_path(cache_dir, prefix, s).exists()
        for s in ("train", "validation", "test")
    )


def build_vqa_embedding_cache(
    *,
    cache_dir: str | Path,
    dataset_id: str,
    clip_model_name: str,
    clip_pretrained: str,
    model,
    splits: dict,
    image_transform,
    device: torch.device,
    batch_size: int = 64,
    max_choices: int = MAX_CHOICES,
) -> None:
    """Precompute frozen CLIP embeddings for every split and save to disk.

    ``model`` must expose ``_encode_images``, ``_encode_texts``, and
    ``embed_dim`` (any VisionQAJEPABase instance works).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = _cache_prefix(dataset_id, clip_model_name, clip_pretrained)

    collate_fn = make_collate_vision_qa(image_transform=image_transform)
    use_amp = device.type == "cuda"
    ctx = RunContext(device=device, use_amp=use_amp)

    for split_name, ds in splits.items():
        path = _split_path(cache_dir, prefix, split_name)
        if path.exists():
            print(f"  VQA cache exists: {path}")
            continue

        print(f"  Precomputing {split_name} embeddings …")
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        all_img, all_q, all_choice = [], [], []
        all_ans, all_nc = [], []
        all_subj: list[str] = []
        all_grade: list[str] = []
        D = model.embed_dim

        for batch in tqdm(loader, desc=f"Precompute {split_name}"):
            batch.images = batch.images.to(device)

            img_emb = model._encode_images(batch.images, ctx)
            q_emb = model._encode_texts(batch.questions, ctx)

            flat_choices = [c if c else " " for choices in batch.choices for c in choices]
            counts = [len(ch) for ch in batch.choices]

            raw = model._encode_texts(flat_choices, ctx)
            raw = raw / (raw.norm(dim=-1, keepdim=True) + 1e-12)

            B = len(batch.choices)
            ce = torch.zeros(B, max_choices, D, device=device)
            offset = 0
            for i, count in enumerate(counts):
                n = min(count, max_choices)
                ce[i, :n] = raw[offset : offset + n]
                offset += count

            all_img.append(img_emb.cpu().half())
            all_q.append(q_emb.cpu().half())
            all_choice.append(ce.cpu().half())
            all_ans.append(batch.answer_indices)
            all_nc.append(batch.num_choices)
            if batch.subject_list:
                all_subj.extend(batch.subject_list)
            if batch.grade_list:
                all_grade.extend(batch.grade_list)

        data = {
            "image_emb": torch.cat(all_img),
            "question_emb": torch.cat(all_q),
            "choice_embs": torch.cat(all_choice),
            "answer_indices": torch.cat(all_ans),
            "num_choices": torch.cat(all_nc),
            "subject_list": all_subj,
            "grade_list": all_grade,
            "embed_dim": D,
            "max_choices": max_choices,
        }
        torch.save(data, path)
        n = data["image_emb"].shape[0]
        print(f"  Saved {split_name} cache: {path} ({n} samples)")

        del data, all_img, all_q, all_choice, all_ans, all_nc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("VQA embedding cache complete.")


def load_vqa_split_cache(
    cache_dir: str | Path,
    dataset_id: str,
    clip_model: str,
    clip_pretrained: str,
    split: str = "train",
) -> dict:
    cache_dir = Path(cache_dir)
    prefix = _cache_prefix(dataset_id, clip_model, clip_pretrained)
    path = _split_path(cache_dir, prefix, split)
    print(f"Loading VQA cache: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


class CachedVQADataset(Dataset):
    """Dataset backed by precomputed CLIP embeddings."""

    def __init__(self, cache_data: dict):
        img = cache_data["image_emb"].float()
        q = cache_data["question_emb"].float()
        self.fused_emb = torch.cat([img, q], dim=1)
        self.choice_embs = cache_data["choice_embs"].float()
        self.answer_indices = cache_data["answer_indices"]
        self.num_choices = cache_data["num_choices"]

    def __len__(self) -> int:
        return len(self.answer_indices)

    def __getitem__(self, idx: int):
        return (
            self.fused_emb[idx],
            self.choice_embs[idx],
            self.answer_indices[idx],
            self.num_choices[idx],
        )


def build_cached_vqa_loaders(
    train_cache: dict,
    val_cache: dict,
    batch_size: int,
):
    train_loader = DataLoader(
        CachedVQADataset(train_cache),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        CachedVQADataset(val_cache),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader
