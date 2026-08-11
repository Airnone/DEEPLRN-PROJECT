from __future__ import annotations

import pytest

from scripts.hparam_sweep import EvalGuard, TrainingHalted


def test_finding_only_guard_does_not_require_ner_or_tuple_signal():
    guard = EvalGuard(
        label="finding-only",
        plateau_metric="finding_macro_f1",
        ner_enabled=False,
    )

    guard({"eval_loss": 1.0, "finding_macro_f1": 0.1, "ner_f1": 0.0})
    guard({"eval_loss": 0.9, "finding_macro_f1": 0.2, "ner_f1": 0.0})
    guard({"eval_loss": 0.8, "finding_macro_f1": 0.3, "ner_f1": 0.0})


def test_finding_only_guard_halts_on_zero_finding_f1():
    guard = EvalGuard(
        label="finding-only",
        plateau_metric="finding_macro_f1",
        ner_enabled=False,
    )

    with pytest.raises(TrainingHalted, match="finding_macro_f1 is 0.0"):
        guard({"eval_loss": 1.0, "finding_macro_f1": 0.0, "ner_f1": 0.0})


def test_guard_plateau_uses_configured_metric():
    guard = EvalGuard(
        label="finding-only",
        plateau_metric="finding_macro_f1",
        ner_enabled=False,
    )
    metrics = {"eval_loss": 1.0, "finding_macro_f1": 0.1, "tuple_f1": 0.0}

    guard(metrics)
    guard(metrics)
    guard(metrics)
    with pytest.raises(TrainingHalted, match="finding_macro_f1 has not improved"):
        guard(metrics)
