"""
Banking77 dataset loader for JEPA text classification.
Loads from Hugging Face (PolyAI/banking77): 77 banking intent classes.
Each sample: (text, label_idx) where label_idx is 0-76.
"""

from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset

# Will be populated from dataset features on first load
LABELS: List[str] = []
NUM_CLASSES: int = 77


def _ensure_labels_loaded() -> None:
    """Load label names from dataset if not yet loaded."""
    global LABELS, NUM_CLASSES
    if LABELS:
        return
    ds = load_dataset("legacy-datasets/banking77", split="train")
    LABELS = list(ds.features["label"].names)  # type: ignore
    NUM_CLASSES = len(LABELS)


def load_banking77(
    cache_dir: str | Path | None = None,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Load Banking77 train and test splits from Hugging Face.
    Downloads automatically on first call.

    Returns:
        (train_samples, test_samples) where each sample is (text, label_idx).
        label_idx is 0-76.
    """
    _ensure_labels_loaded()
    cache = str(cache_dir) if cache_dir else None
    ds = load_dataset("legacy-datasets/banking77", cache_dir=cache)
    train = [(ex["text"], ex["label"]) for ex in ds["train"]]
    test = [(ex["text"], ex["label"]) for ex in ds["test"]]
    return train, test


class Banking77Dataset:
    """PyTorch-friendly dataset that yields (text, label_idx) for DataLoader."""

    def __init__(self, samples: List[Tuple[str, int]], label_to_idx: dict[str, int] | None = None):
        self.samples = samples
        _ensure_labels_loaded()
        self.label_to_idx = label_to_idx or get_label_to_idx()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        text, label_idx = self.samples[idx]
        return text, label_idx


def get_labels() -> List[str]:
    """Return list of label names (loads from dataset if needed)."""
    _ensure_labels_loaded()
    return LABELS


def get_label_to_idx() -> dict[str, int]:
    """Return mapping from label name to 0-indexed class index."""
    _ensure_labels_loaded()
    return {label: i for i, label in enumerate(LABELS)}


# Backward-compatible alias for code that expects load_ag_news
def load_dataset_splits(
    data_dir: str | Path | None = None,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Load train and test splits. Uses data_dir as cache_dir if provided."""
    return load_banking77(cache_dir=data_dir)


if __name__ == "__main__":
    train, test = load_banking77()
    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    print(f"Num classes: {NUM_CLASSES}")
    print(f"Example: text={train[0][0][:80]}... label_idx={train[0][1]} -> {LABELS[train[0][1]]}")
