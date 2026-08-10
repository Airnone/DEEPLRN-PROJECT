"""Dependency-free evaluation metrics for the DEEPLRN research tasks.

The functions in this module operate on plain Python collections so the same
definitions can be reused by training, command-line evaluation, and tests.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

import torch

from deeplrn.config import NER_CFG


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = _divide(true_positive, true_positive + false_positive)
    recall = _divide(true_positive, true_positive + false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _divide(2 * precision * recall, precision + recall),
    }


def set_prf(predicted: Iterable[Hashable], gold: Iterable[Hashable]) -> dict[str, float]:
    """Exact-match precision, recall, and F1 for two sets of structured items."""

    predicted_set = set(predicted)
    gold_set = set(gold)
    counts = _prf(
        len(predicted_set & gold_set),
        len(predicted_set - gold_set),
        len(gold_set - predicted_set),
    )
    counts.update(
        {
            "support": float(len(gold_set)),
            "predicted": float(len(predicted_set)),
        }
    )
    return counts


def _tag_parts(tag: str) -> tuple[str, str | None]:
    if tag == "O" or "-" not in tag:
        return "O", None
    prefix, entity_type = tag.split("-", 1)
    return prefix, entity_type


def bio_spans_from_offsets(
    tag_ids: Sequence[int],
    token_offsets: Sequence[Sequence[int]],
    tags: Sequence[str] | None = None,
) -> set[tuple[str, int, int]]:
    """Decode BIO tags to exact global character spans.

    Invalid ``I`` transitions are treated as a new entity. Special and padded
    tokens use negative offsets and close any active span.
    """

    tag_names = list(tags or NER_CFG.tags)
    spans: set[tuple[str, int, int]] = set()
    active_type: str | None = None
    active_start = -1
    active_end = -1

    def close() -> None:
        nonlocal active_type, active_start, active_end
        if active_type is not None and active_start >= 0 and active_end > active_start:
            spans.add((active_type, active_start, active_end))
        active_type = None
        active_start = -1
        active_end = -1

    for tag_id, offset in zip(tag_ids, token_offsets):
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end <= start or int(tag_id) < 0 or int(tag_id) >= len(tag_names):
            close()
            continue

        prefix, entity_type = _tag_parts(tag_names[int(tag_id)])
        if prefix == "B" or (prefix == "I" and entity_type != active_type):
            close()
            active_type = entity_type
            active_start = start
            active_end = end
        elif prefix == "I" and active_type == entity_type:
            active_end = max(active_end, end)
        else:
            close()

    close()
    return spans


def entity_span_metrics(
    predicted_by_document: dict[str, set[tuple[str, int, int]]],
    gold_by_document: dict[str, set[tuple[str, int, int]]],
) -> dict[str, float]:
    """Micro exact-match entity metrics, keeping document identity distinct."""

    predicted = {
        (doc_id, *span)
        for doc_id, spans in predicted_by_document.items()
        for span in spans
    }
    gold = {
        (doc_id, *span)
        for doc_id, spans in gold_by_document.items()
        for span in spans
    }
    return set_prf(predicted, gold)


def classification_metrics(
    predictions: Sequence[int], gold: Sequence[int], num_classes: int
) -> dict[str, float]:
    """Accuracy and macro P/R/F1 with zero-safe absent-class handling."""

    if len(predictions) != len(gold):
        raise ValueError("predictions and gold labels must have the same length")
    if not gold:
        return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    per_class = []
    for class_id in range(num_classes):
        tp = sum(p == class_id and g == class_id for p, g in zip(predictions, gold))
        fp = sum(p == class_id and g != class_id for p, g in zip(predictions, gold))
        fn = sum(p != class_id and g == class_id for p, g in zip(predictions, gold))
        per_class.append(_prf(tp, fp, fn))

    return {
        "accuracy": _divide(sum(p == g for p, g in zip(predictions, gold)), len(gold)),
        "macro_precision": sum(item["precision"] for item in per_class) / num_classes,
        "macro_recall": sum(item["recall"] for item in per_class) / num_classes,
        "macro_f1": sum(item["f1"] for item in per_class) / num_classes,
    }


def multilabel_classification_metrics(
    predictions: Sequence[Sequence[int | float]],
    gold: Sequence[Sequence[int | float]],
    num_labels: int,
) -> dict[str, float]:
    """Subset accuracy and macro/micro P/R/F1 for binary label vectors."""

    if len(predictions) != len(gold):
        raise ValueError("predictions and gold labels must have the same length")
    if num_labels < 1:
        raise ValueError("num_labels must be positive")
    if not gold:
        return {
            "subset_accuracy": 0.0,
            "hamming_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
        }
    predicted_rows = [[int(bool(value)) for value in row] for row in predictions]
    gold_rows = [[int(bool(value)) for value in row] for row in gold]
    if any(len(row) != num_labels for row in predicted_rows + gold_rows):
        raise ValueError("every label vector must match num_labels")

    per_label = []
    total_tp = total_fp = total_fn = 0
    for label_id in range(num_labels):
        tp = sum(
            predicted[label_id] == 1 and target[label_id] == 1
            for predicted, target in zip(predicted_rows, gold_rows)
        )
        fp = sum(
            predicted[label_id] == 1 and target[label_id] == 0
            for predicted, target in zip(predicted_rows, gold_rows)
        )
        fn = sum(
            predicted[label_id] == 0 and target[label_id] == 1
            for predicted, target in zip(predicted_rows, gold_rows)
        )
        per_label.append(_prf(tp, fp, fn))
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro = _prf(total_tp, total_fp, total_fn)
    result = {
        "subset_accuracy": _divide(
            sum(predicted == target for predicted, target in zip(predicted_rows, gold_rows)),
            len(gold_rows),
        ),
        "hamming_accuracy": _divide(
            sum(
                predicted == target
                for predicted_row, gold_row in zip(predicted_rows, gold_rows)
                for predicted, target in zip(predicted_row, gold_row)
            ),
            len(gold_rows) * num_labels,
        ),
        "macro_precision": sum(item["precision"] for item in per_label) / num_labels,
        "macro_recall": sum(item["recall"] for item in per_label) / num_labels,
        "macro_f1": sum(item["f1"] for item in per_label) / num_labels,
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
    }
    for label_id, item in enumerate(per_label):
        result[f"label_{label_id}_precision"] = item["precision"]
        result[f"label_{label_id}_recall"] = item["recall"]
        result[f"label_{label_id}_f1"] = item["f1"]
    return result


def calibration_metrics(
    probabilities: torch.Tensor, gold: Sequence[int], num_bins: int = 10
) -> dict[str, float]:
    """Multiclass Brier score and top-label expected calibration error."""

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (examples, classes)")
    if probabilities.shape[0] != len(gold):
        raise ValueError("probability rows and gold labels must have the same length")
    if not gold:
        return {"brier": 0.0, "ece": 0.0}

    probs = probabilities.detach().float().cpu()
    targets = torch.zeros_like(probs)
    targets[torch.arange(len(gold)), torch.tensor(gold, dtype=torch.long)] = 1.0
    brier = torch.square(probs - targets).sum(dim=1).mean().item()

    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(torch.tensor(gold, dtype=torch.long)).float()
    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (confidence > lower) & (confidence <= upper)
        if index == 0:
            in_bin |= confidence.eq(0)
        if in_bin.any():
            weight = in_bin.float().mean().item()
            ece += weight * abs(
                correct[in_bin].mean().item() - confidence[in_bin].mean().item()
            )
    return {"brier": float(brier), "ece": float(ece)}


def multilabel_calibration_metrics(
    probabilities: torch.Tensor,
    gold: Sequence[Sequence[int | float]],
    num_bins: int = 10,
) -> dict[str, float]:
    """Binary Brier score and ECE over all example-label decisions."""

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (examples, labels)")
    if probabilities.shape[0] != len(gold):
        raise ValueError("probability rows and gold labels must have the same length")
    if not gold:
        return {"brier": 0.0, "ece": 0.0}
    targets = torch.tensor(gold, dtype=torch.float)
    probs = probabilities.detach().float().cpu()
    if targets.shape != probs.shape:
        raise ValueError("gold label vectors must match probability shape")
    flat_probs = probs.flatten()
    flat_targets = targets.flatten()
    brier = torch.square(flat_probs - flat_targets).mean().item()
    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    for index in range(num_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (flat_probs > lower) & (flat_probs <= upper)
        if index == 0:
            in_bin |= flat_probs.eq(0)
        if in_bin.any():
            ece += in_bin.float().mean().item() * abs(
                flat_targets[in_bin].mean().item() - flat_probs[in_bin].mean().item()
            )
    return {"brier": float(brier), "ece": float(ece)}


def relation_metrics(
    predictions: Sequence[int],
    gold: Sequence[int],
    *,
    no_relation_id: int,
) -> dict[str, float]:
    """Micro P/R/F1 over positive relation types.

    A wrong positive type is both a false positive and a false negative. True
    ``NO_RELATION`` decisions do not inflate the score on negative-heavy data.
    """

    if len(predictions) != len(gold):
        raise ValueError("predictions and gold labels must have the same length")
    tp = fp = fn = 0
    for predicted, target in zip(predictions, gold):
        if predicted == target and target != no_relation_id:
            tp += 1
        else:
            if predicted != no_relation_id:
                fp += 1
            if target != no_relation_id:
                fn += 1
    result = _prf(tp, fp, fn)
    result["support"] = float(sum(item != no_relation_id for item in gold))
    return result


def relation_tuples_from_candidates(
    doc_ids: Sequence[str],
    relation_triples: Sequence[Sequence[Sequence[int]]],
    token_offsets: torch.Tensor,
    labels: Sequence[int],
    *,
    no_relation_id: int,
) -> set[tuple]:
    """Build provenance-aware relation tuples from flattened candidate labels."""

    tuples: set[tuple] = set()
    label_index = 0
    offsets = token_offsets.detach().cpu()
    for doc_index, triples in enumerate(relation_triples):
        for triple in triples:
            if label_index >= len(labels):
                raise ValueError("fewer relation labels than relation candidates")
            label = int(labels[label_index])
            label_index += 1
            if label == no_relation_id:
                continue
            hc, hs, he, tc, ts, te, _ = (int(value) for value in triple)
            h_start = int(offsets[doc_index, hc, hs, 0])
            h_end = int(offsets[doc_index, hc, he - 1, 1])
            t_start = int(offsets[doc_index, tc, ts, 0])
            t_end = int(offsets[doc_index, tc, te - 1, 1])
            if min(h_start, h_end, t_start, t_end) < 0:
                continue
            tuples.add((doc_ids[doc_index], h_start, h_end, t_start, t_end, label))
    if label_index != len(labels):
        raise ValueError("more relation labels than relation candidates")
    return tuples
