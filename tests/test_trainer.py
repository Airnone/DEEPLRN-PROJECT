import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deeplrn.training.dataset import DeepLRNDataset
from deeplrn.training.trainer import SilentMLBugError, Trainer, TrainingConfig


class DeterministicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            ner_loss_weight=1.0,
            cls_loss_weight=0.5,
            rel_loss_weight=0.5,
        )

    def forward(self, input_ids, relation_triples, **_):
        batch, chunks, tokens = input_ids.shape
        ner_logits = torch.zeros(batch, chunks, tokens, 17, device=input_ids.device)
        ner_logits[..., 0] = self.scale
        cls_logits = torch.zeros(batch, 10, device=input_ids.device)
        cls_logits[..., 0] = self.scale
        relation_count = sum(len(items) for items in relation_triples)
        rel_logits = torch.zeros(relation_count, 4, device=input_ids.device)
        if relation_count:
            rel_logits[..., 1] = self.scale
        return {
            "ner_logits": ner_logits,
            "cls_logits": cls_logits,
            "rel_logits": rel_logits if relation_count else None,
            "loss": self.scale.square(),
        }


class NonFiniteLossModel(DeterministicModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        outputs["loss"] = self.scale * torch.tensor(float("nan"))
        return outputs


class PlateauLossModel(DeterministicModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        outputs["loss"] = self.scale * 0.0 + 1.0
        return outputs


def _write_record(path):
    path.write_text(
        json.dumps(
            {
                "doc_id": "doc",
                "finding_label_id": 0,
                "chunks": [
                    {
                        "input_ids": [2, 3, 4, 5],
                        "attention_mask": [1, 1, 1, 1],
                        "ner_labels": [0, 0, 0, 0],
                        "token_offsets": [[0, 1], [1, 2], [2, 3], [3, 4]],
                    }
                ],
                "relation_triples": [[0, 0, 1, 0, 2, 3, 1]],
            }
        ),
        encoding="utf-8",
    )


def test_trainer_computes_metrics_and_saves_resumable_final_checkpoint(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    _write_record(records / "doc.json")
    dataset = DeepLRNDataset(records, max_tokens=4)
    output = tmp_path / "checkpoints"
    trainer = Trainer(
        DeterministicModel(),
        dataset,
        dataset,
        TrainingConfig(
            num_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            output_dir=str(output),
            eval_every=0,
            save_every=0,
            log_every=0,
        ),
    )

    initial = trainer.evaluate()
    assert initial["finding_accuracy"] == 1.0
    assert initial["relation_f1"] == 1.0
    assert initial["tuple_f1"] == 1.0
    trainer.train()
    assert (output / "best.pt").exists()
    assert (output / "final.pt").exists()

    resumed = Trainer(
        DeterministicModel(),
        dataset,
        dataset,
        TrainingConfig(num_epochs=1, batch_size=1, output_dir=str(output)),
    )
    checkpoint = resumed.load_checkpoint(output / "final.pt")
    assert checkpoint["epoch"] == 1
    assert checkpoint["global_step"] == 1


def test_trainer_stops_on_non_finite_loss(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    _write_record(records / "doc.json")
    dataset = DeepLRNDataset(records, max_tokens=4)
    trainer = Trainer(
        NonFiniteLossModel(),
        dataset,
        dataset,
        TrainingConfig(
            num_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            output_dir=str(tmp_path / "checkpoints"),
            eval_every=0,
            save_every=0,
            log_every=0,
        ),
    )

    with pytest.raises(SilentMLBugError, match="non-finite training loss"):
        trainer.train()


def test_trainer_stops_after_epoch_when_ner_f1_is_zero(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    _write_record(records / "doc.json")
    record_path = records / "doc.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["chunks"][0]["ner_labels"] = [1, 2, 0, 0]
    record_path.write_text(json.dumps(record), encoding="utf-8")
    dataset = DeepLRNDataset(records, max_tokens=4)
    trainer = Trainer(
        DeterministicModel(),
        dataset,
        dataset,
        TrainingConfig(
            num_epochs=2,
            batch_size=1,
            gradient_accumulation_steps=1,
            output_dir=str(tmp_path / "checkpoints"),
            eval_every=0,
            save_every=0,
            log_every=0,
        ),
    )

    with pytest.raises(SilentMLBugError, match="validation NER F1 is 0.0 after epoch 1"):
        trainer.train()

    assert trainer.completed_epochs == 1


def test_trainer_stops_when_first_epoch_loss_plateaus(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    for index in range(4):
        _write_record(records / f"doc-{index}.json")
    dataset = DeepLRNDataset(records, max_tokens=4)
    trainer = Trainer(
        PlateauLossModel(),
        dataset,
        dataset,
        TrainingConfig(
            num_epochs=2,
            batch_size=1,
            gradient_accumulation_steps=1,
            output_dir=str(tmp_path / "checkpoints"),
            eval_every=0,
            save_every=0,
            log_every=0,
        ),
    )

    with pytest.raises(SilentMLBugError, match="loss did not improve during the first epoch"):
        trainer.train()

    assert trainer.completed_epochs == 1


def test_trainer_rejects_validation_label_absent_from_training(tmp_path):
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    train_dir.mkdir()
    validation_dir.mkdir()
    _write_record(train_dir / "train.json")
    _write_record(validation_dir / "validation.json")
    validation_path = validation_dir / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["finding_label_id"] = 2
    validation_path.write_text(json.dumps(validation), encoding="utf-8")

    with pytest.raises(ValueError, match="absent from training"):
        Trainer(
            DeterministicModel(),
            DeepLRNDataset(train_dir, max_tokens=4),
            DeepLRNDataset(validation_dir, max_tokens=4),
            TrainingConfig(output_dir=str(tmp_path / "checkpoints")),
        )


def test_trainer_requires_opt_in_for_machine_draft_labels(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    _write_record(records / "doc.json")
    path = records / "doc.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["annotation_metadata"] = {"supervision_quality": "machine_draft"}
    path.write_text(json.dumps(record), encoding="utf-8")
    dataset = DeepLRNDataset(records, max_tokens=4)

    with pytest.raises(ValueError, match="machine-draft"):
        Trainer(
            DeterministicModel(),
            dataset,
            dataset,
            TrainingConfig(output_dir=str(tmp_path / "checkpoints")),
        )
