import pytest
import torch

from deeplrn.evaluation import (
    bio_spans_from_offsets,
    calibration_metrics,
    classification_metrics,
    relation_metrics,
    relation_tuples_from_candidates,
    set_prf,
)


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

    calibration = calibration_metrics(
        torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]), [0, 1]
    )
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
    tuples = relation_tuples_from_candidates(
        ["doc"], triples, offsets, [1], no_relation_id=0
    )
    assert tuples == {("doc", 10, 15, 20, 25, 1)}
    assert set_prf(tuples, tuples)["f1"] == 1.0
