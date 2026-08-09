"""Guarded learning-rate x batch-size sweep for DEEPLRN.

This script exists so that the moment a real leakage-controlled split
manifest is available (``deeplrn split`` output), a hyperparameter sweep can
be launched with a single command while enforcing the project's research
guardrails in code rather than relying on someone watching logs:

* Only ``train`` and ``validation`` splits are ever loaded. The ``test``
  split name never appears anywhere in this script, and the manifest is
  checked up front so a test-only manifest is refused outright.
* Each evaluation is inspected by a guard callback. A NaN/Inf loss, or a
  0.0 finding-F1 and 0.0 NER-F1 at the same time after a full epoch, halts
  the run immediately (``STOP EXECUTION IMMEDIATELY`` per the project's
  guardrails) instead of continuing to burn compute on a broken run.
* A halted grid cell stops the whole sweep rather than silently moving on
  to the next combination.
* Nothing here ever deletes ``output/``, checkpoints, or logs.

Usage
-----
    python scripts/hparam_sweep.py --manifest manifests/split.json

    # Narrow the grid
    python scripts/hparam_sweep.py --manifest manifests/split.json \\
        --learning-rates 1e-5 2e-5 --batch-sizes 4 8 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deeplrn.model.deeplrn_model import DeepLRNModel, ModelConfig
from deeplrn.training.dataset import DeepLRNDataset
from deeplrn.training.trainer import Trainer, TrainingConfig

logger = logging.getLogger("deeplrn.hparam_sweep")

DEFAULT_LEARNING_RATES = (1e-5, 2e-5, 3e-5, 5e-5)
DEFAULT_BATCH_SIZES = (4, 8, 16)
PLATEAU_PATIENCE = 3
PLATEAU_EPSILON = 1e-4


class TrainingHalted(RuntimeError):
    """Raised by the guard callback to stop a run on a silent ML bug."""


@dataclass
class EvalGuard:
    """Watches evaluation metrics and halts training on a silent ML bug."""

    label: str
    epoch_hint: int = 0
    _seen_evals: int = 0
    _best_for_plateau: float = float("-inf")
    _stale_evals: int = 0

    def __call__(self, metrics: dict[str, float]) -> None:
        self._seen_evals += 1

        loss = metrics.get("eval_loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            raise TrainingHalted(
                f"[{self.label}] eval_loss is {loss} (NaN/Inf) at evaluation "
                f"#{self._seen_evals} -- halting for human judgment."
            )

        finding_f1 = metrics.get("finding_macro_f1", 0.0)
        ner_f1 = metrics.get("ner_f1", 0.0)
        if finding_f1 == 0.0 and ner_f1 == 0.0:
            raise TrainingHalted(
                f"[{self.label}] finding_macro_f1 and ner_f1 are both 0.0 at "
                f"evaluation #{self._seen_evals} -- the model appears to be "
                f"learning nothing. Halting for human judgment."
            )

        score = metrics.get("tuple_f1", 0.0)
        if score > self._best_for_plateau + PLATEAU_EPSILON:
            self._best_for_plateau = score
            self._stale_evals = 0
        else:
            self._stale_evals += 1
        if self._stale_evals >= PLATEAU_PATIENCE:
            raise TrainingHalted(
                f"[{self.label}] tuple_f1 has not improved by more than "
                f"{PLATEAU_EPSILON} for {self._stale_evals} consecutive "
                f"evaluations -- training appears plateaued. Halting for "
                f"human judgment."
            )


def _load_manifest_splits(manifest_path: Path) -> set[str]:
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    return {document.get("split") for document in manifest.get("documents", [])}


def _require_trainable_manifest(manifest_path: Path) -> None:
    splits = _load_manifest_splits(manifest_path)
    if "train" not in splits or "validation" not in splits:
        raise SystemExit(
            f"Refusing to run: manifest {manifest_path} does not contain "
            f"both a 'train' and a 'validation' split (found: {sorted(s for s in splits if s)}). "
            "This sweep never trains against the 'test' split, and there is "
            "nothing else it is authorized to use."
        )


def run_sweep(
    manifest_path: Path,
    *,
    learning_rates: tuple[float, ...],
    batch_sizes: tuple[int, ...],
    epochs: int,
    seed: int,
    output_root: Path,
    results_path: Path,
) -> list[dict[str, Any]]:
    _require_trainable_manifest(manifest_path)

    train_dataset = DeepLRNDataset.from_manifest(manifest_path, "train")
    eval_dataset = DeepLRNDataset.from_manifest(manifest_path, "validation")
    if len(train_dataset) == 0:
        raise SystemExit("Refusing to run: the 'train' split is empty.")
    if len(eval_dataset) == 0:
        raise SystemExit("Refusing to run: the 'validation' split is empty.")

    results: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            label = f"lr{learning_rate}_bs{batch_size}"
            run_dir = output_root / label
            logger.info("=== starting grid cell %s -> %s ===", label, run_dir)

            model = DeepLRNModel(ModelConfig())
            config = TrainingConfig(
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=epochs,
                seed=seed,
                output_dir=str(run_dir),
            )
            guard = EvalGuard(label=label)
            trainer = Trainer(
                model, train_dataset, eval_dataset, config, on_evaluate=guard
            )

            entry: dict[str, Any] = {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "output_dir": str(run_dir),
                "status": "completed",
            }
            try:
                metrics = trainer.train()
                entry["metrics"] = metrics
                entry["finding_macro_f1"] = metrics.get("finding_macro_f1")
                entry["tuple_f1"] = metrics.get("tuple_f1")
                entry["best_checkpoint"] = str(run_dir / "best.pt")
            except TrainingHalted as halt:
                entry["status"] = "halted"
                entry["halt_reason"] = str(halt)
                entry["metrics"] = trainer.last_metrics
                entry["best_checkpoint"] = str(run_dir / "best.pt")
                results.append(entry)
                results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                logger.error(
                    "Sweep stopped: grid cell %s halted on a silent ML bug:\n%s",
                    label,
                    halt,
                )
                print(json.dumps(results, indent=2))
                raise SystemExit(
                    f"\nSweep halted at grid cell {label} -- see reason above. "
                    "Not continuing to remaining grid cells; this needs human "
                    "review before further runs."
                ) from halt

            results.append(entry)
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            logger.info(
                "grid cell %s complete: finding_macro_f1=%s tuple_f1=%s",
                label,
                entry.get("finding_macro_f1"),
                entry.get("tuple_f1"),
            )

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=list(DEFAULT_LEARNING_RATES)
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES)
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints/sweep"))
    parser.add_argument(
        "--results", type=Path, default=Path("sweeps/sweep_results.json")
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    results = run_sweep(
        args.manifest,
        learning_rates=tuple(args.learning_rates),
        batch_sizes=tuple(args.batch_sizes),
        epochs=args.epochs,
        seed=args.seed,
        output_root=args.output_root,
        results_path=args.results,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
