import json
from types import SimpleNamespace

import torch
from torch import nn

from deeplrn.training.dataset import DeepLRNDataset
from deeplrn.training.trainer import Trainer, TrainingConfig


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
        cls_logits = torch.zeros(batch, 5, device=input_ids.device)
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

