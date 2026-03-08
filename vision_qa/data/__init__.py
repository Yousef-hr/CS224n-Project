"""Vision QA data registry, dataset wrapper, and collate."""

from pathlib import Path
from typing import Any, Callable

import torch
from datasets import load_dataset

from vision_qa.base import VisionQABatch

from .science_qa import science_qa_get_item, science_qa_get_labels

vision_qa_registry: dict[str, dict[str, Any]] = {
    "science_qa": {
        "id": "derek-thomas/ScienceQA",
        "subset": None,
        "get_item": science_qa_get_item,
        "get_labels": science_qa_get_labels,
    },
}


class VisionQADataset:
    """Wraps an HF split and yields (image, question, choices, answer_idx) or + (subject, grade) per item."""

    def __init__(
        self,
        split: Any,
        get_item_fn: Callable[..., tuple[Any, ...]],
        use_image: bool = True,
    ):
        self.split = split
        self._use_image = use_image
        self._get_item = get_item_fn

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        ex = self.split[idx]
        if callable(self._get_item) and self._get_item.__code__.co_argcount >= 2:
            return self._get_item(ex, use_image=self._use_image)
        return self._get_item(ex)


def make_collate_vision_qa(
    image_transform: Callable[[Any], torch.Tensor] | None = None,
    placeholder_shape: tuple[int, ...] = (3, 224, 224),
) -> Callable[[list], VisionQABatch]:
    """
    Returns a collate function that builds VisionQABatch.
    If image_transform is provided, images are converted to tensor [B, C, H, W] (placeholder zeros for None).
    If not provided, images are left as list of PIL|None (model must handle conversion).
    """

    def collate(batch: list[tuple[Any, ...]]) -> VisionQABatch:
        # Support 4-tuple (image, q, choices, ans) or 6-tuple (+ subject, grade)
        has_meta = len(batch[0]) >= 6
        images_raw = [b[0] for b in batch]
        questions = [b[1] for b in batch]
        choices_list = [b[2] for b in batch]
        answer_indices = torch.tensor([b[3] for b in batch], dtype=torch.long)
        subject_list = [b[4] for b in batch] if has_meta else None
        grade_list = [b[5] for b in batch] if has_meta else None

        max_choices = max(len(c) for c in choices_list)
        num_choices_per = torch.tensor([len(c) for c in choices_list], dtype=torch.long)
        choices_padded = [c + [""] * (max_choices - len(c)) for c in choices_list]

        if image_transform is not None:
            images_tensor = []
            for img in images_raw:
                if img is not None and hasattr(img, "convert"):
                    images_tensor.append(image_transform(img))
                else:
                    images_tensor.append(torch.zeros(placeholder_shape, dtype=torch.float32))
            images = torch.stack(images_tensor)
        else:
            images = torch.zeros(len(batch), *placeholder_shape, dtype=torch.float32)

        return VisionQABatch(
            images=images,
            questions=questions,
            choices=choices_padded,
            answer_indices=answer_indices,
            num_choices=num_choices_per,
            subject_list=subject_list,
            grade_list=grade_list,
        )

    return collate


def load_vision_qa(
    name: str,
    subset: str | None = None,
    cache_dir: str | Path | None = None,
    use_image: bool = True,
) -> tuple[VisionQADataset, VisionQADataset, VisionQADataset, Callable[[Any], list[str]]]:
    """
    Load train, validation, and test splits for a registered vision QA dataset.
    Returns (train_ds, val_ds, test_ds, get_labels).
    """
    if name not in vision_qa_registry:
        raise KeyError(f"Unknown dataset: {name}. Registered: {list(vision_qa_registry.keys())}")

    recipe = vision_qa_registry[name]
    cache = str(cache_dir) if cache_dir is not None else None
    dataset_id = recipe["id"]
    subset = subset or recipe.get("subset")
    get_item = recipe["get_item"]
    get_labels = recipe["get_labels"]

    if subset is not None:
        ds_dict = load_dataset(dataset_id, subset, cache_dir=cache)
    else:
        ds_dict = load_dataset(dataset_id, cache_dir=cache)

    train_ds = VisionQADataset(ds_dict["train"], get_item, use_image=use_image)
    val_ds = VisionQADataset(ds_dict["validation"], get_item, use_image=use_image)
    test_ds = VisionQADataset(ds_dict["test"], get_item, use_image=use_image)

    def _get_labels(split: Any) -> list[str]:
        return get_labels(split)

    return train_ds, val_ds, test_ds, _get_labels
