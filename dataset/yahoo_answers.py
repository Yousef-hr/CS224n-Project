"""
Yahoo Answers Topics dataset helpers.
"""

from pathlib import Path
from typing import Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

DATASET_NAME = "yahoo_answers_topics"


def _cache_dir_str(cache_dir: str | Path | None) -> str | None:
    return str(cache_dir) if cache_dir is not None else None


def _compose_text(example: dict) -> str:
    title = (example.get("question_title") or "").strip()
    body = (example.get("question_content") or "").strip()
    answer = (example.get("best_answer") or "").strip()
    return " ".join(part for part in (title, body, answer) if part)


def load_yahoo_answers_dataset(cache_dir: str | Path | None = None):
    """Load Yahoo Answers Topics DatasetDict with train/test splits."""
    return load_dataset(DATASET_NAME, cache_dir=_cache_dir_str(cache_dir))


def get_labels(cache_dir: str | Path | None = None) -> list[str]:
    """Return ordered topic names."""
    ds = load_yahoo_answers_dataset(cache_dir=cache_dir)
    return list(ds["train"].features["topic"].names)  # type: ignore[attr-defined]


def load_yahoo_answers(
    cache_dir: str | Path | None = None,
) -> Tuple[list[Tuple[str, int]], list[Tuple[str, int]]]:
    """Return train/test splits as Python lists of `(text, label_idx)`."""
    ds = load_yahoo_answers_dataset(cache_dir=cache_dir)
    train = [(_compose_text(ex), int(ex["topic"])) for ex in ds["train"]]
    test = [(_compose_text(ex), int(ex["topic"])) for ex in ds["test"]]
    return train, test


class YahooAnswersDataset(Dataset):
    """PyTorch dataset yielding `(text, label_idx)` from a HF split."""

    def __init__(self, split):
        self.split = split

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        ex = self.split[idx]
        return _compose_text(ex), int(ex["topic"])


if __name__ == "__main__":
    ds_dict = load_yahoo_answers_dataset()
    labels = get_labels()
    print(f"Train samples: {len(ds_dict['train'])}")
    print(f"Test samples: {len(ds_dict['test'])}")
    print(f"Num classes: {len(labels)}")
    sample_text, sample_label = YahooAnswersDataset(ds_dict["train"])[0]
    print(f"Example: text={sample_text[:100]}... label_idx={sample_label} -> {labels[sample_label]}")
