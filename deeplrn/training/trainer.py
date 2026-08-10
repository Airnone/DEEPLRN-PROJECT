"""Reproducible multi-task training and evaluation."""

from __future__ import annotations

import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from deeplrn.checkpoints import build_checkpoint, load_checkpoint, save_checkpoint
from deeplrn.evaluation import (
    bio_spans_from_offsets,
    entity_span_metrics,
    multilabel_calibration_metrics,
    multilabel_classification_metrics,
    relation_metrics,
    relation_tuples_from_candidates,
    set_prf,
)
from deeplrn.schema import FINDING_LABELS

from .dataset import collate_documents

logger = logging.getLogger(__name__)

_PLATEAU_MIN_BATCHES = 4
_PLATEAU_MIN_RELATIVE_LOSS_DROP = 1e-3


class SilentMLBugError(RuntimeError):
    """Raised when training exhibits a failure mode that requires human review."""


def finding_counts(dataset) -> torch.Tensor:
    if dataset is None or len(dataset) == 0:
        return torch.zeros(len(FINDING_LABELS), dtype=torch.long)
    return torch.stack(
        [dataset[index].finding_labels for index in range(len(dataset))]
    ).sum(dim=0).long()


def validate_finding_coverage(train_dataset, validation_dataset) -> None:
    """Reject validation labels the model never sees in training."""

    if validation_dataset is None:
        return
    training_counts = finding_counts(train_dataset)
    validation_counts = finding_counts(validation_dataset)
    missing = [
        FINDING_LABELS[index]
        for index in range(len(FINDING_LABELS))
        if validation_counts[index] > 0 and training_counts[index] == 0
    ]
    if missing:
        raise ValueError(
            "validation contains finding labels absent from training: "
            f"{missing}; expand/adjudicate the corpus or choose a predeclared "
            "leakage-safe split before training"
        )


def validate_supervision_quality(
    train_dataset, validation_dataset, *, allow_weak_supervision: bool
) -> None:
    """Require an explicit opt-in before training on machine-draft labels."""

    weak_documents = []
    for dataset in (train_dataset, validation_dataset):
        if dataset is None:
            continue
        weak_documents.extend(
            dataset[index].doc_id
            for index in range(len(dataset))
            if dataset[index].metadata.get("supervision_quality") == "machine_draft"
        )
    if weak_documents and not allow_weak_supervision:
        raise ValueError(
            f"{len(weak_documents)} train/validation observations use machine-draft "
            "labels; adjudicate them or explicitly enable weak supervision"
        )


def _select_device() -> torch.device:
    """Pick the best available accelerator: CUDA, then Apple Silicon MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    num_workers: int = 0

    ner_loss_weight: float = 1.0
    cls_loss_weight: float = 0.5
    rel_loss_weight: float = 0.5

    output_dir: str = "checkpoints"
    log_every: int = 10
    eval_every: int = 50
    save_every: int = 100
    selection_metric: str = "tuple_f1"
    finding_threshold: float = 0.5
    balance_finding_labels: bool = True
    allow_weak_supervision: bool = False


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        eval_dataset=None,
        config: TrainingConfig | None = None,
        on_evaluate: Callable[[dict[str, float]], None] | None = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainingConfig()
        self.on_evaluate = on_evaluate
        if self.config.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if not 0.0 <= self.config.finding_threshold <= 1.0:
            raise ValueError("finding_threshold must be between zero and one")
        validate_finding_coverage(self.train_dataset, self.eval_dataset)
        validate_supervision_quality(
            self.train_dataset,
            self.eval_dataset,
            allow_weak_supervision=self.config.allow_weak_supervision,
        )

        self._set_seed(self.config.seed)
        self.device = _select_device()
        self.model.to(self.device)
        self._configure_finding_balance()

        # Keep the model's actual joint loss aligned with the run manifest.
        for name in ("ner_loss_weight", "cls_loss_weight", "rel_loss_weight"):
            if hasattr(self.model, "config"):
                setattr(self.model.config, name, getattr(self.config, name))

        generator = torch.Generator().manual_seed(self.config.seed)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_documents,
            num_workers=self.config.num_workers,
            generator=generator,
        )
        self.eval_loader = (
            DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_documents,
                num_workers=self.config.num_workers,
            )
            if self.eval_dataset is not None
            else None
        )

        self.optimizer, self.scheduler = self._setup_optimizers()
        self.global_step = 0
        self.completed_epochs = 0
        self.last_metrics: dict[str, float] = {}
        self.best_score = float("-inf")
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _configure_finding_balance(self) -> None:
        classifier = getattr(self.model, "classifier", None)
        if (
            not self.config.balance_finding_labels
            or classifier is None
            or not hasattr(classifier, "set_pos_weight")
        ):
            return
        targets = [
            self.train_dataset[index].finding_labels
            for index in range(len(self.train_dataset))
        ]
        if not targets:
            return
        target_tensor = torch.stack(targets).float()
        positives = target_tensor.sum(dim=0)
        negatives = len(target_tensor) - positives
        weights = torch.where(
            positives > 0,
            (negatives / positives.clamp(min=1.0)).clamp(min=1.0, max=20.0),
            torch.ones_like(positives),
        )
        classifier.set_pos_weight(weights)
        logger.info(
            "Finding positive counts=%s; BCE positive weights=%s",
            positives.int().tolist(),
            [round(float(value), 3) for value in weights],
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if not any(item in name for item in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if any(item in name for item in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.config.learning_rate
        )
        updates_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.config.gradient_accumulation_steps)
        )
        total_steps = max(1, updates_per_epoch * self.config.num_epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps,
        )
        return optimizer, scheduler

    def _forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            chunk_mask=batch["chunk_mask"].to(self.device),
            ner_labels=batch["ner_labels"].to(self.device),
            finding_labels=batch["finding_labels"].to(self.device),
            relation_triples=batch["relation_triples"],
            page_ids=batch["page_ids"].to(self.device),
            section_ids=batch["section_ids"].to(self.device),
            bboxes=batch["bboxes"].to(self.device),
        )

    def train(self) -> dict[str, float]:
        """Train, evaluate at configured intervals, and save best/final checkpoints."""

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.completed_epochs, self.config.num_epochs):
            epoch_losses: list[float] = []
            iterator = tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
            )
            for step, batch in enumerate(iterator):
                outputs = self._forward(batch)
                if "loss" not in outputs:
                    raise RuntimeError("model returned no loss for a labelled training batch")
                raw_loss = outputs["loss"]
                loss_value = float(raw_loss.detach().item())
                if not math.isfinite(loss_value):
                    raise SilentMLBugError(
                        f"non-finite training loss at epoch {epoch + 1}, batch {step + 1}: "
                        f"{loss_value}; training stopped for human review"
                    )
                epoch_losses.append(loss_value)
                loss = raw_loss / self.config.gradient_accumulation_steps
                loss.backward()

                last_batch = step + 1 == len(self.train_loader)
                should_update = (
                    (step + 1) % self.config.gradient_accumulation_steps == 0 or last_batch
                )
                if not should_update:
                    continue

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.config.log_every and self.global_step % self.config.log_every == 0:
                    logger.info(
                        "Step %d - loss %.4f",
                        self.global_step,
                        loss_value,
                    )
                if (
                    self.eval_loader is not None
                    and self.config.eval_every
                    and self.global_step % self.config.eval_every == 0
                ):
                    self._evaluate_and_maybe_save_best()
                    self.model.train()
                if self.config.save_every and self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(
                        os.path.join(
                            self.config.output_dir, f"checkpoint-{self.global_step}.pt"
                        )
                    )

            self.completed_epochs = epoch + 1
            if self.eval_loader is not None:
                metrics = self._evaluate_and_maybe_save_best()
                average_loss = sum(epoch_losses) / len(epoch_losses)
                relative_loss_drop = self._relative_loss_drop(epoch_losses)
                logger.info(
                    "Epoch %d - train_loss %.4f - validation_macro_f1 %.4f "
                    "- validation_tuple_f1 %.4f - validation_ner_f1 %.4f",
                    self.completed_epochs,
                    average_loss,
                    metrics["finding_macro_f1"],
                    metrics["tuple_f1"],
                    metrics["ner_f1"],
                )
                self._check_epoch_for_silent_bugs(
                    epoch=epoch,
                    metrics=metrics,
                    relative_loss_drop=relative_loss_drop,
                )
                self.model.train()

        final_path = os.path.join(self.config.output_dir, "final.pt")
        self.save_checkpoint(final_path)
        return dict(self.last_metrics)

    @staticmethod
    def _relative_loss_drop(losses: list[float]) -> float | None:
        if len(losses) < _PLATEAU_MIN_BATCHES:
            return None
        window = max(1, len(losses) // 4)
        initial = sum(losses[:window]) / window
        final = sum(losses[-window:]) / window
        return (initial - final) / max(abs(initial), 1e-12)

    @staticmethod
    def _check_epoch_for_silent_bugs(
        *,
        epoch: int,
        metrics: dict[str, float],
        relative_loss_drop: float | None,
    ) -> None:
        if (
            epoch == 0
            and relative_loss_drop is not None
            and relative_loss_drop < _PLATEAU_MIN_RELATIVE_LOSS_DROP
        ):
            raise SilentMLBugError(
                "training loss did not improve during the first epoch "
                f"(relative drop={relative_loss_drop:.6f}); training stopped for human review"
            )
        if metrics.get("ner_gold_support", 0.0) > 0 and metrics["ner_f1"] == 0.0:
            raise SilentMLBugError(
                f"validation NER F1 is 0.0 after epoch {epoch + 1}; "
                "training stopped for human review"
            )

    def _evaluate_and_maybe_save_best(self) -> dict[str, float]:
        metrics = self.evaluate()
        self.last_metrics = metrics
        logger.info("Evaluation at step %d: %s", self.global_step, metrics)
        if self.config.selection_metric not in metrics:
            raise KeyError(
                f"selection metric {self.config.selection_metric!r} is unavailable"
            )
        score = metrics[self.config.selection_metric]
        if score > self.best_score:
            self.best_score = score
            self.save_checkpoint(
                os.path.join(self.config.output_dir, "best.pt"), metrics=metrics
            )
        # Checkpoint is already on disk before this runs, so a caller that
        # raises here (e.g. to halt on NaN loss or a stuck 0.0 F1) never
        # loses the best result found so far.
        if self.on_evaluate is not None:
            self.on_evaluate(metrics)
        return metrics

    def evaluate(self) -> dict[str, float]:
        """Evaluate exact entity spans, findings, relations, tuples, and calibration."""

        if self.eval_loader is None:
            raise ValueError("evaluate requires an evaluation dataset")
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        predicted_entities: dict[str, set] = defaultdict(set)
        gold_entities: dict[str, set] = defaultdict(set)
        finding_probabilities = []
        finding_predictions: list[list[int]] = []
        finding_gold: list[list[float]] = []
        relation_predictions: list[int] = []
        relation_gold: list[int] = []
        predicted_tuples: set[tuple] = set()
        gold_tuples: set[tuple] = set()
        evidence_page_correct = 0
        evidence_page_total = 0
        # Relation records reserve class 0 for NO_RELATION; positive schema
        # labels are numbered from 1 in TrainingRecordBuilder and inference.
        no_relation_id = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                outputs = self._forward(batch)
                if outputs.get("loss") is not None:
                    total_loss += float(outputs["loss"].item())
                batch_count += 1

                ner_predictions = outputs["ner_logits"].argmax(dim=-1).cpu()
                ner_gold = batch["ner_labels"].cpu()
                offsets = batch["token_offsets"].cpu()
                for doc_index, doc_id in enumerate(batch["doc_ids"]):
                    chunk_count = int(batch["chunk_mask"][doc_index].sum().item())
                    for chunk_index in range(chunk_count):
                        chunk_offsets = offsets[doc_index, chunk_index].tolist()
                        predicted_entities[doc_id].update(
                            bio_spans_from_offsets(
                                ner_predictions[doc_index, chunk_index].tolist(),
                                chunk_offsets,
                            )
                        )
                        gold_entities[doc_id].update(
                            bio_spans_from_offsets(
                                ner_gold[doc_index, chunk_index].tolist(), chunk_offsets
                            )
                        )

                probabilities = outputs["cls_logits"].sigmoid().cpu()
                gold_batch = batch["finding_labels"].tolist()
                finding_probabilities.append(probabilities)
                finding_predictions.extend(
                    (probabilities > self.config.finding_threshold).int().tolist()
                )
                finding_gold.extend(gold_batch)

                rel_logits = outputs.get("rel_logits")
                if rel_logits is not None:
                    predicted_batch = rel_logits.argmax(dim=-1).cpu().tolist()
                    gold_batch_rel = [
                        int(triple[-1])
                        for triples in batch["relation_triples"]
                        for triple in triples
                    ]
                    relation_predictions.extend(predicted_batch)
                    relation_gold.extend(gold_batch_rel)
                    candidate_index = 0
                    page_ids = batch["page_ids"].cpu()
                    for doc_index, triples in enumerate(batch["relation_triples"]):
                        metadata_items = batch["metadata"][doc_index].get(
                            "relation_candidate_metadata", []
                        )
                        for local_index, triple in enumerate(triples):
                            predicted_label = predicted_batch[candidate_index]
                            gold_label = gold_batch_rel[candidate_index]
                            candidate_index += 1
                            if (
                                predicted_label != gold_label
                                or gold_label == no_relation_id
                                or local_index >= len(metadata_items)
                            ):
                                continue
                            expected_pages = set(
                                metadata_items[local_index].get(
                                    "evidence_page_numbers", []
                                )
                            )
                            if not expected_pages:
                                continue
                            hc, hs, he, tc, ts, te, _ = (
                                int(value) for value in triple
                            )
                            predicted_pages = {
                                int(value)
                                for value in torch.cat(
                                    (
                                        page_ids[doc_index, hc, hs:he],
                                        page_ids[doc_index, tc, ts:te],
                                    )
                                ).tolist()
                                if int(value) > 0
                            }
                            evidence_page_total += 1
                            evidence_page_correct += predicted_pages == expected_pages
                    predicted_tuples.update(
                        relation_tuples_from_candidates(
                            batch["doc_ids"],
                            batch["relation_triples"],
                            offsets,
                            predicted_batch,
                            no_relation_id=no_relation_id,
                        )
                    )
                    gold_tuples.update(
                        relation_tuples_from_candidates(
                            batch["doc_ids"],
                            batch["relation_triples"],
                            offsets,
                            gold_batch_rel,
                            no_relation_id=no_relation_id,
                        )
                    )

        entity = entity_span_metrics(predicted_entities, gold_entities)
        ner_gold_support = sum(len(spans) for spans in gold_entities.values())
        finding = multilabel_classification_metrics(
            finding_predictions, finding_gold, len(FINDING_LABELS)
        )
        probability_tensor = (
            torch.cat(finding_probabilities, dim=0)
            if finding_probabilities
            else torch.empty((0, len(FINDING_LABELS)))
        )
        calibration = multilabel_calibration_metrics(probability_tensor, finding_gold)
        relation = relation_metrics(
            relation_predictions,
            relation_gold,
            no_relation_id=no_relation_id,
        )
        tuples = set_prf(predicted_tuples, gold_tuples)
        metrics = {
            "eval_loss": total_loss / batch_count if batch_count else 0.0,
            "ner_precision": entity["precision"],
            "ner_recall": entity["recall"],
            "ner_f1": entity["f1"],
            "ner_gold_support": float(ner_gold_support),
            "finding_subset_accuracy": finding["subset_accuracy"],
            "finding_accuracy": finding["subset_accuracy"],
            "finding_hamming_accuracy": finding["hamming_accuracy"],
            "finding_macro_precision": finding["macro_precision"],
            "finding_macro_recall": finding["macro_recall"],
            "finding_macro_f1": finding["macro_f1"],
            "finding_micro_precision": finding["micro_precision"],
            "finding_micro_recall": finding["micro_recall"],
            "finding_micro_f1": finding["micro_f1"],
            "finding_brier": calibration["brier"],
            "finding_ece": calibration["ece"],
            "relation_precision": relation["precision"],
            "relation_recall": relation["recall"],
            "relation_f1": relation["f1"],
            "tuple_precision": tuples["precision"],
            "tuple_recall": tuples["recall"],
            "tuple_f1": tuples["f1"],
            "evidence_page_accuracy": (
                evidence_page_correct / evidence_page_total
                if evidence_page_total
                else 0.0
            ),
            "evidence_page_support": float(evidence_page_total),
        }
        for label_id, label in enumerate(FINDING_LABELS):
            for measure in ("precision", "recall", "f1"):
                metrics[f"finding_{label}_{measure}"] = finding[
                    f"label_{label_id}_{measure}"
                ]
        return metrics

    def save_checkpoint(
        self, path: str, *, metrics: dict[str, float] | None = None
    ) -> None:
        checkpoint_metrics = dict(
            metrics if metrics is not None else self.last_metrics
        )
        if self.best_score != float("-inf"):
            checkpoint_metrics["best_selection_score"] = self.best_score
        checkpoint = build_checkpoint(
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            training_config=self.config,
            epoch=self.completed_epochs,
            global_step=self.global_step,
            metrics=checkpoint_metrics,
        )
        save_checkpoint(path, checkpoint)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        checkpoint = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.completed_epochs = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))
        self.last_metrics = dict(checkpoint.get("metrics", {}))
        if "best_selection_score" in self.last_metrics:
            self.best_score = self.last_metrics["best_selection_score"]
        elif self.config.selection_metric in self.last_metrics:
            self.best_score = self.last_metrics[self.config.selection_metric]
        logger.info("Checkpoint loaded from %s", path)
        return checkpoint
