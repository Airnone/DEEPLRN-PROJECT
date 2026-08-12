import pytest
import torch

from deeplrn.evaluation import (
    apply_multilabel_thresholds,
    bio_spans_from_offsets,
    calibration_metrics,
    classification_metrics,
    multilabel_calibration_metrics,
    multilabel_classification_metrics,
    relation_metrics,
    relation_tuples_from_candidates,
    set_prf,
    tune_multilabel_thresholds,
)


def test_multilabel_metrics_score_exact_and_partial_predictions():
    predictions = [[1, 0, 1], [0, 1, 0]]
    gold = [[1, 0, 1], [1, 1, 0]]
    metrics = multilabel_classification_metrics(predictions, gold, 3)
    assert metrics["subset_accuracy"] == 0.5
    assert metrics["micro_f1"] == pytest.approx(6 / 7)
    calibration = multilabel_calibration_metrics(
        torch.tensor([[0.9, 0.1, 0.8], [0.4, 0.7, 0.2]]), gold
    )
    assert 0.0 <= calibration["brier"] <= 1.0


def test_multilabel_metrics_report_support_aware_macro_f1():
    predictions = [[1, 0, 1], [0, 0, 0]]
    gold = [[1, 0, 1], [0, 0, 0]]

    metrics = multilabel_classification_metrics(predictions, gold, 3)

    assert metrics["macro_f1"] == pytest.approx(2 / 3)
    assert metrics["supported_label_count"] == 2.0
    assert metrics["supported_macro_f1"] == 1.0
    assert metrics["label_0_support"] == 1.0
    assert metrics["label_1_support"] == 0.0
    assert metrics["label_2_support"] == 1.0


def test_per_label_threshold_tuning_improves_f1_and_keeps_unsupported_default():
    probabilities = [[0.40, 0.90], [0.30, 0.80], [0.20, 0.70]]
    gold = [[1, 0], [1, 0], [0, 0]]

    thresholds = tune_multilabel_thresholds(probabilities, gold, 2)
    predictions = apply_multilabel_thresholds(probabilities, thresholds)
    metrics = multilabel_classification_metrics(predictions, gold, 2)

    assert thresholds[0] == pytest.approx(0.2)
    assert thresholds[1] == pytest.approx(0.5)
    assert metrics["label_0_f1"] == pytest.approx(1.0)


def test_bio_spans_decode_global_offsets_and_recover_invalid_i():
    tags = ["O", "B-PERSON", "I-PERSON", "B-AMOUNT", "I-AMOUNT"]
    spans = bio_spans_from_offsets(
        [0, 1, 2, 0, 4],
        [[-1, -1], [10, 14], [15, 18], [19, 20], [21, 24]],
        tags,
    )
    assert spans == {("PERSON", 10, 18), ("AMOUNT", 21, 24)}


def test_classification_and_calibration_metrics_are_real_values():
    metrics = classification_metrics([0, 1, 1, 2], [0, 1, 2, 2], 3)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert 0.0 < metrics["macro_f1"] < 1.0

    calibration = calibration_metrics(torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]), [0, 1])
    assert calibration["brier"] == pytest.approx(0.1)
    assert calibration["ece"] == pytest.approx(0.25)


def test_relation_f1_excludes_true_no_relation_examples():
    # 0 is NO_RELATION. One true positive, one missed positive, one false positive.
    metrics = relation_metrics([1, 0, 2, 0], [1, 2, 0, 0], no_relation_id=0)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_provenance_relation_tuples_use_character_spans():
    offsets = torch.tensor([[[[10, 12], [12, 15], [20, 22], [22, 25]]]])
    triples = [[(0, 0, 2, 0, 2, 4, 1)]]
    tuples = relation_tuples_from_candidates(["doc"], triples, offsets, [1], no_relation_id=0)
    assert tuples == {("doc", 10, 15, 20, 25, 1)}
    assert set_prf(tuples, tuples)["f1"] == 1.0
