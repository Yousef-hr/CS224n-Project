"""
Banking77 dataset helpers.
"""

from datasets import Dataset

def __get_banking77_item__(ex: any):
    return ex["text"], int(ex["label"])

def __get_banking77_labels__(ds: Dataset) -> list[str]:
    return list(ds.features["label"].names)  # type: ignore[attr-defined]

banking77_dataset_registry_dict = {
    "id": "legacy-datasets/banking77",
    "subset": None,
    "get_item": __get_banking77_item__,
    "get_labels": __get_banking77_labels__,
}