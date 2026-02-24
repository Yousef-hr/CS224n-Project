"""
CLINC OOS dataset helpers.
"""

from pathlib import Path
from typing import Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

DATASET_NAME = "clinc_oos"


def _cache_dir_str(cache_dir: str | Path | None) -> str | None:
    return str(cache_dir) if cache_dir is not None else None


def load_clinc_oos_dataset(config: str = "plus", cache_dir: str | Path | None = None):
    """Load CLINC OOS DatasetDict with train/validation/test splits."""
    return load_dataset(DATASET_NAME, config, cache_dir=_cache_dir_str(cache_dir))


def get_labels(config: str = "plus", cache_dir: str | Path | None = None) -> list[str]:
    """Return ordered intent names."""
    ds = load_clinc_oos_dataset(config=config, cache_dir=cache_dir)
    return list(ds["train"].features["intent"].names)  # type: ignore[attr-defined]


def load_clinc_oos(
    config: str = "plus",
    cache_dir: str | Path | None = None,
) -> Tuple[list[Tuple[str, int]], list[Tuple[str, int]], list[Tuple[str, int]]]:
    """Return train/validation/test splits as Python lists."""
    ds = load_clinc_oos_dataset(config=config, cache_dir=cache_dir)
    train = [(ex["text"], int(ex["intent"])) for ex in ds["train"]]
    validation = [(ex["text"], int(ex["intent"])) for ex in ds["validation"]]
    test = [(ex["text"], int(ex["intent"])) for ex in ds["test"]]
    return train, validation, test


class CLINCOOSDataset(Dataset):
    """PyTorch dataset yielding `(text, label_idx)` from a HF split."""

    def __init__(self, split):
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        ex = self.split[idx]
        return ex["text"], int(ex["intent"])

