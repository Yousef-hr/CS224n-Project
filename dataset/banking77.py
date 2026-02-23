"""
Minimal Banking77 dataset helpers.
"""

from pathlib import Path
from typing import Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

DATASET_NAME = "legacy-datasets/banking77"


def _cache_dir_str(cache_dir: str | Path | None) -> str | None:
    return str(cache_dir) if cache_dir is not None else None


def load_banking77_dataset(cache_dir: str | Path | None = None):
    """Load Banking77 Hugging Face DatasetDict with `train` and `test` splits."""
    return load_dataset(DATASET_NAME, cache_dir=_cache_dir_str(cache_dir))


def get_labels(cache_dir: str | Path | None = None) -> list[str]:
    """Return ordered label names."""
    ds = load_banking77_dataset(cache_dir=cache_dir)
    return list(ds["train"].features["label"].names)  # type: ignore[attr-defined]


def get_label_to_idx(cache_dir: str | Path | None = None) -> dict[str, int]:
    """Return mapping from label name to class index."""
    labels = get_labels(cache_dir=cache_dir)
    return {label: i for i, label in enumerate(labels)}


def load_banking77(
    cache_dir: str | Path | None = None,
) -> Tuple[list[Tuple[str, int]], list[Tuple[str, int]]]:
    """Compatibility helper returning train/test as Python lists."""
    ds = load_banking77_dataset(cache_dir=cache_dir)
    train = [(ex["text"], int(ex["label"])) for ex in ds["train"]]
    test = [(ex["text"], int(ex["label"])) for ex in ds["test"]]
    return train, test


class Banking77Dataset(Dataset):
    """PyTorch dataset yielding `(text, label_idx)` from a HF split."""

    def __init__(self, split):
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        ex = self.split[idx]
        return ex["text"], int(ex["label"])


def load_dataset_splits(
    data_dir: str | Path | None = None,
) -> Tuple[list[Tuple[str, int]], list[Tuple[str, int]]]:
    """Backward-compatible alias using `data_dir` as cache dir."""
    return load_banking77(cache_dir=data_dir)


if __name__ == "__main__":
    train, test = load_banking77()
    labels = get_labels()
    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    print(f"Num classes: {len(labels)}")
    print(f"Example: text={train[0][0][:80]}... label_idx={train[0][1]} -> {labels[train[0][1]]}")
