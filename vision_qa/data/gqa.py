"""
GQA dataset: load from HuggingFace lmms-lab/GQA (instructions + images configs).

GQA is open-ended (one answer per question). We convert to multiple-choice by
sampling 4 distractor answers from the train vocabulary; choices = [correct] + 4
wrong, then shuffled. Same task as ScienceQA for the models (pick correct among 5).

Uses the balanced split only (train_balanced_instructions ~943k).
"""

from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset

NUM_CHOICES = 5  # 1 correct + 4 distractors; matches SOTA max_choices and cache.
GQA_DATASET_ID = "lmms-lab/GQA"

# Balanced configs: instructions (question/answer) and images.
CONFIGS = {
    "train": "train_balanced_instructions",
    "validation": "val_balanced_instructions",
    "test": "test_balanced_instructions",
}
IMAGE_CONFIGS = {
    "train": "train_balanced_images",
    "validation": "val_balanced_images",
    "test": "test_balanced_images",
}


def _get_answer(ex: dict) -> str:
    """Extract single answer string from a GQA example."""
    a = ex.get("answer") or ex.get("answers")
    if a is None:
        return ""
    if isinstance(a, list):
        return str(a[0]) if a else ""
    return str(a).strip()


def _get_question(ex: dict) -> str:
    return (ex.get("question") or ex.get("question_text") or "").strip()


def _get_image_id(ex: dict) -> str | int:
    """Id used to join with images split (image_id or id)."""
    return ex.get("image_id") or ex.get("imageId") or ex.get("id")


def load_gqa(
    cache_dir: str | None = None,
    use_image: bool = True,
    num_choices: int = NUM_CHOICES,
    distractor_seed: int = 42,
) -> tuple[Any, Any, Any, list[str]]:
    """
    Load GQA train/val/test (balanced split only, ~943k train).
    Builds answer vocabulary from train and forms multiple-choice with distractors.
    Returns (train_ds, val_ds, test_ds, get_labels) where each ds yields (image, question, choices, answer_idx).
    """
    from . import VisionQADataset  # avoid circular import: only when load_gqa is called

    cache = str(cache_dir) if cache_dir else None
    configs = CONFIGS
    image_configs = IMAGE_CONFIGS

    def _get_split(ds, split_name: str = "train"):
        """Return the Dataset for the given split; handle DatasetDict or single Dataset."""
        if hasattr(ds, "get") and split_name in ds:
            return ds[split_name]
        if hasattr(ds, "values"):
            return list(ds.values())[0]
        return ds

    # Load train instructions first to build vocabulary.
    _train_raw = load_dataset(
        GQA_DATASET_ID,
        configs["train"],
        cache_dir=cache,
        trust_remote_code=True,
    )
    train_instr = _get_split(_train_raw, "train")
    vocab: set[str] = set()
    for i in range(len(train_instr)):
        a = _get_answer(train_instr[i])
        if a:
            vocab.add(a)
    vocab_list = sorted(vocab)
    print(f"GQA train answer vocabulary size: {len(vocab_list)}")

    # Load images for each split and build id -> index (for merge).
    def load_images(split_key: str):
        config = image_configs[split_key]
        raw = load_dataset(
            GQA_DATASET_ID,
            config,
            cache_dir=cache,
            trust_remote_code=True,
        )
        ds = _get_split(raw, "train")
        id_to_idx = {}
        for idx in range(len(ds)):
            row = ds[idx]
            key = row.get("id") or row.get("image_id")
            if key is not None:
                id_to_idx[str(key)] = idx
        return ds, id_to_idx

    train_img_ds, train_id_to_idx = load_images("train")
    val_img_ds, val_id_to_idx = load_images("validation")
    test_img_ds, test_id_to_idx = load_images("test")

    rng = random.Random(distractor_seed)

    def make_get_item(images_ds, id_to_idx: dict):
        def get_item(ex: dict, use_image: bool = True) -> tuple[Any, str, list[str], int]:
            image = None
            if use_image and images_ds is not None:
                key = _get_image_id(ex)
                if key is not None:
                    idx = id_to_idx.get(str(key))
                    if idx is not None:
                        row = images_ds[idx]
                        image = row.get("image")
            question = _get_question(ex)
            answer = _get_answer(ex)
            if not answer or answer not in vocab_list:
                answer = vocab_list[0] if vocab_list else "yes"
            others = [a for a in vocab_list if a != answer]
            if len(others) >= num_choices - 1:
                distractors = rng.sample(others, num_choices - 1)
            else:
                distractors = others + [answer] * (num_choices - 1 - len(others))
                distractors = distractors[: num_choices - 1]
            choices = [answer] + distractors
            rng.shuffle(choices)
            answer_idx = choices.index(answer)
            return (image, question, choices, answer_idx)

        return get_item

    _val_raw = load_dataset(
        GQA_DATASET_ID,
        configs["validation"],
        cache_dir=cache,
        trust_remote_code=True,
    )
    val_instr = _get_split(_val_raw, "train")
    _test_raw = load_dataset(
        GQA_DATASET_ID,
        configs["test"],
        cache_dir=cache,
        trust_remote_code=True,
    )
    test_instr = _get_split(_test_raw, "train")

    train_ds = VisionQADataset(
        train_instr,
        make_get_item(train_img_ds, train_id_to_idx),
        use_image=use_image,
    )
    val_ds = VisionQADataset(
        val_instr,
        make_get_item(val_img_ds, val_id_to_idx),
        use_image=use_image,
    )
    test_ds = VisionQADataset(
        test_instr,
        make_get_item(test_img_ds, test_id_to_idx),
        use_image=use_image,
    )

    def get_labels(_: Any) -> list[str]:
        return [f"choice_{i}" for i in range(num_choices)]

    return train_ds, val_ds, test_ds, get_labels
