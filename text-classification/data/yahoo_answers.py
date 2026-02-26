"""
Yahoo Answers Topics dataset helpers.
"""

from datasets import Dataset    

def __compose_yahoo_input__(example: dict) -> str:
    title = (example.get("question_title") or "").strip()
    body = (example.get("question_content") or "").strip()
    answer = (example.get("best_answer") or "").strip()
    return " ".join(part for part in (title, body, answer) if part)

def __get_yahoo_item__(ex: any):
    return __compose_yahoo_input__(ex), int(ex["topic"])

def __get_yahoo_labels__(ds: Dataset) -> list[str]:
    return list(ds.features["topic"].names) 

yahoo_answers_dataset_registry_dict = {
    "id": "yahoo_answers_topics",
    "subset": None,
    "get_item": __get_yahoo_item__,
    "get_labels": __get_yahoo_labels__,
}