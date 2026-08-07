import json

from deeplrn.experiments import get_preset, train_tfidf_svm


def test_neural_presets_define_distinct_context_and_layout_controls():
    assert get_preset("deeplrn").model_options == {}
    assert get_preset("sentence_roberta").model_options["use_document_context"] is False
    assert get_preset("sentence_roberta").single_sentence_chunks is True
    assert get_preset("longformer").max_tokens == 4096
    assert get_preset("layoutlmv3").model_options["encoder_uses_2d_positions"] is True


def test_tfidf_svm_baseline_trains_and_persists_metrics(tmp_path):
    labels = [
        "unauthorized_expenditure",
        "unliquidated_cash_advance",
    ]
    documents = []
    for split in ("train", "validation"):
        for index, label in enumerate(labels):
            path = tmp_path / f"{split}-{index}.json"
            path.write_text(
                json.dumps(
                    {
                        "doc_id": path.stem,
                        "document_text": "unauthorized purchase" if index == 0 else "cash advance",
                        "finding_label": label,
                    }
                ),
                encoding="utf-8",
            )
            documents.append({"split": split, "source_path": path.name})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": documents}), encoding="utf-8")
    output = tmp_path / "baseline.joblib"

    metrics = train_tfidf_svm(manifest, output, max_features=100)
    assert metrics["accuracy"] == 1.0
    assert output.exists()
    assert output.with_suffix(".metrics.json").exists()
