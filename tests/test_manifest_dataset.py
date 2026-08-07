import json

from deeplrn.training.dataset import DeepLRNDataset


def test_dataset_selects_only_requested_manifest_split(tmp_path):
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    train.write_text(json.dumps({"doc_id": "train", "chunks": []}), encoding="utf-8")
    validation.write_text(
        json.dumps({"doc_id": "validation", "chunks": []}), encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "documents": [
                    {"split": "train", "source_path": train.name},
                    {"split": "validation", "source_path": validation.name},
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = DeepLRNDataset.from_manifest(manifest, "validation", max_tokens=8)
    assert len(dataset) == 1
    assert dataset[0].doc_id == "validation"

