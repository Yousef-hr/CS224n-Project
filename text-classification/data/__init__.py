from pathlib import Path
from typing import Tuple

from datasets import load_dataset

from .banking77 import banking77_dataset_registry_dict
from .clinc_oos import clinc_oos_dataset_registry_dict
from .yahoo_answers import yahoo_answers_dataset_registry_dict

text_classification_registry = {
    "banking77": banking77_dataset_registry_dict,
    "clinc_oos": clinc_oos_dataset_registry_dict,
    "yahoo_answers": yahoo_answers_dataset_registry_dict,
}

class TextClassificationDataset:
    """Yields (text: str, label_idx: int) from a HuggingFace split."""

    def __init__(self, split, get_item_fn):
        self.split = split
        self.get_item_fn = get_item_fn

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        ex = self.split[idx]
        return self.get_item_fn(ex)


def load_text_classification(
    name: str,
    subset: str | None = None,
    cache_dir: str | Path | None = None,
):
    """Load train/test splits and label list for a registered text classification dataset.
    Returns (train_ds, test_ds, get_labels).
    For datasets with a config (e.g. clinc_oos), pass config to override the registry default (e.g. "plus", "small", "imbalanced").
    """
    recipe = text_classification_registry[name]
    cache = str(cache_dir) if cache_dir is not None else None

    dataset_id = recipe["id"]
    subset = subset if subset is not None else recipe["subset"]
    get_item = recipe["get_item"]
    get_labels = recipe["get_labels"]

    if subset is not None:
        ds_dict = load_dataset(dataset_id, subset, cache_dir=cache)
    else:
        ds_dict = load_dataset(dataset_id, cache_dir=cache)

    train_split = TextClassificationDataset(ds_dict["train"], get_item)
    test_split = TextClassificationDataset(ds_dict["test"], get_item)

    return train_split, test_split, get_labels