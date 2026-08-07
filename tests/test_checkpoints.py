from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from deeplrn.checkpoints import build_checkpoint, validate_checkpoint


@dataclass
class TinyConfig:
    width: int = 2


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TinyConfig()
        self.layer = torch.nn.Linear(2, 2)


def test_checkpoint_contains_model_and_label_contracts():
    checkpoint = build_checkpoint(TinyModel(), global_step=7)
    validate_checkpoint(checkpoint)
    assert checkpoint["model_config"] == {"width": 2}
    assert checkpoint["global_step"] == 7
    assert "ASSOCIATED_WITH" in checkpoint["label_schema"]["relation_types"]


def test_checkpoint_rejects_wrong_label_order():
    checkpoint = build_checkpoint(TinyModel())
    checkpoint["label_schema"]["relation_types"].reverse()
    with pytest.raises(ValueError, match="label schema"):
        validate_checkpoint(checkpoint)
