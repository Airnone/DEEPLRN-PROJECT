"""Versioned checkpoint helpers shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from deeplrn.schema import label_schema


CHECKPOINT_FORMAT_VERSION = 2


def _config_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"cannot serialize configuration of type {type(value).__name__}")


def build_checkpoint(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    training_config: Any = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Build a self-describing checkpoint dictionary.

    Model configuration and label order are part of the checkpoint contract;
    omitting either can silently map trained logits to the wrong labels.
    """

    if not hasattr(model, "config"):
        raise ValueError("model must expose a serializable .config attribute")

    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "model_config": _config_dict(model.config),
        "training_config": _config_dict(training_config),
        "label_schema": label_schema(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics or {}),
    }


def validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    version = int(checkpoint.get("checkpoint_format_version", 0))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {version}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    required = {"model_state_dict", "model_config", "label_schema"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {sorted(missing)}")
    if checkpoint["label_schema"] != label_schema():
        raise ValueError("checkpoint label schema does not match this DEEPLRN version")


def save_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(checkpoint), target)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    validate_checkpoint(checkpoint)
    return checkpoint
