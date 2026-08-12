from scripts.build_disagreement_review_queue import select_review_queue


def _candidate(doc_id: str, split: str, label: str, score: float):
    return {
        "doc_id": doc_id,
        "split": split,
        "implicated_labels": [label],
        "priority_score": score,
    }


def test_queue_is_unique_and_caps_validation_records():
    candidates = [
        _candidate("validation-a", "validation", "unauthorized_expenditure", 10),
        _candidate("validation-b", "validation", "procurement_irregularity", 9),
        _candidate("train-a", "train", "unauthorized_expenditure", 8),
        _candidate("train-b", "train", "procurement_irregularity", 7),
    ]

    selected = select_review_queue(candidates, limit=3, validation_cap=1, per_label_seed=1)

    assert len(selected) == 3
    assert len({item["doc_id"] for item in selected}) == 3
    assert sum(item["split"] == "validation" for item in selected) == 1
