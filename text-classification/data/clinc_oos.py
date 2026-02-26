"""
CLINC OOS dataset helpers.
"""

from datasets import Dataset

def __get_clinc_oos_item__(ex: any):
    return ex["text"], int(ex["intent"])

def __get_clinc_oos_labels__(ds: Dataset) -> list[str]:
    return list(ds.features["intent"].names)

clinc_oos_dataset_registry_dict = {
    "id": "clinc_oos",
    "subset": "plus",
    "get_item": __get_clinc_oos_item__,
    "get_labels": __get_clinc_oos_labels__,
}