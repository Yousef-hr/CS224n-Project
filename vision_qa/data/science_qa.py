"""
ScienceQA dataset helpers.
Load from Hugging Face: derek-thomas/ScienceQA.
Some rows have no image (image column is None).
"""

from typing import Any

from datasets import Dataset


def science_qa_get_item(
    ex: dict[str, Any], use_image: bool = True
) -> tuple[Any, str, list[str], int, str, str]:
    """
    Return (image, question, choices, answer_idx, subject, grade).
    image is PIL.Image or None if missing or use_image=False.
    """
    image = ex.get("image")
    if not use_image:
        image = None
    elif image is not None and hasattr(image, "convert"):
        image = image  # already PIL when accessed from HF
    else:
        image = None  # missing image

    question = ex.get("question", "")
    choices = ex.get("choices") or []
    if isinstance(choices, str):
        choices = [choices]
    answer = ex.get("answer", 0)
    answer_idx = int(answer) if answer is not None else 0
    subject = str(ex.get("subject", ""))
    grade = str(ex.get("grade", ""))
    return image, question, list(choices), answer_idx, subject, grade


def science_qa_get_labels(_ds: Dataset) -> list[str]:
    """Placeholder for registry compatibility; ScienceQA has variable num_choices."""
    return [f"choice_{i}" for i in range(5)]


science_qa_registry_dict = {
    "id": "derek-thomas/ScienceQA",
    "subset": None,
    "get_item": science_qa_get_item,
    "get_labels": science_qa_get_labels,
}
