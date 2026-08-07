import pytest

from deeplrn.statistics import (
    cluster_bootstrap_f1,
    holm_adjust,
    paired_cluster_permutation_test,
)


def test_cluster_bootstrap_is_deterministic_and_cluster_based():
    counts = {"lgu-a": (3, 1, 1), "lgu-b": (2, 0, 2), "lgu-c": (1, 1, 0)}
    first = cluster_bootstrap_f1(counts, iterations=200, seed=7)
    second = cluster_bootstrap_f1(counts, iterations=200, seed=7)
    assert first == second
    assert first["ci_lower"] <= first["f1"] <= first["ci_upper"]


def test_paired_permutation_and_holm_correction():
    better = {"a": (5, 0, 0), "b": (4, 1, 0), "c": (5, 0, 1)}
    worse = {"a": (2, 3, 3), "b": (2, 2, 2), "c": (1, 4, 4)}
    result = paired_cluster_permutation_test(better, worse, iterations=300, seed=1)
    assert result["f1_difference"] > 0
    assert 0 < result["p_value"] <= 1
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

