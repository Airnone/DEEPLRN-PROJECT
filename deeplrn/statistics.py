"""LGU-clustered uncertainty and paired comparison utilities."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

Counts = tuple[int, int, int]  # true positive, false positive, false negative


def _f1(counts: Sequence[Counts]) -> float:
    tp = sum(item[0] for item in counts)
    fp = sum(item[1] for item in counts)
    fn = sum(item[2] for item in counts)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def cluster_bootstrap_f1(
    counts_by_lgu: Mapping[str, Counts],
    *,
    iterations: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """F1 and a percentile interval from resampling whole LGUs."""

    if not counts_by_lgu:
        raise ValueError("at least one LGU cluster is required")
    if iterations < 1 or not 0 < confidence < 1:
        raise ValueError("iterations must be positive and confidence must be in (0, 1)")
    clusters = list(counts_by_lgu)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        selected = [counts_by_lgu[rng.choice(clusters)] for _ in clusters]
        samples.append(_f1(selected))
    samples.sort()
    tail = (1 - confidence) / 2
    lower_index = min(iterations - 1, max(0, int(tail * iterations)))
    upper_index = min(iterations - 1, max(0, int((1 - tail) * iterations) - 1))
    return {
        "f1": _f1(list(counts_by_lgu.values())),
        "ci_lower": samples[lower_index],
        "ci_upper": samples[upper_index],
        "confidence": confidence,
    }


def paired_cluster_permutation_test(
    first: Mapping[str, Counts],
    second: Mapping[str, Counts],
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Two-sided paired randomization test for an F1 difference by LGU."""

    if set(first) != set(second) or not first:
        raise ValueError("both systems must contain the same non-empty LGU set")
    clusters = sorted(first)
    observed = _f1([first[key] for key in clusters]) - _f1(
        [second[key] for key in clusters]
    )
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        permuted_first = []
        permuted_second = []
        for key in clusters:
            left, right = first[key], second[key]
            if rng.random() < 0.5:
                left, right = right, left
            permuted_first.append(left)
            permuted_second.append(right)
        delta = _f1(permuted_first) - _f1(permuted_second)
        extreme += abs(delta) >= abs(observed)
    return {
        "f1_difference": observed,
        "p_value": (extreme + 1) / (iterations + 1),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in the original order."""

    count = len(p_values)
    if any(value < 0 or value > 1 for value in p_values):
        raise ValueError("p-values must be between zero and one")
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted
