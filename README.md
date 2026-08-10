# DEEPLRN

DEEPLRN is an evidence-linked, document-level NLP system for Philippine
Commission on Audit reports. It extracts neutral audit findings, entities,
amounts, projects, and textual relationships. Its output is a review aid, not a
determination of misconduct, intent, liability, or guilt.

The repository is **software-complete and training-ready**. The included pilot
overlays are machine drafts, not human-adjudicated gold labels, and must not be
used for final model-quality claims. It does not contain a trained research
checkpoint or measured held-out results. See [DEVELOPMENT.md](DEVELOPMENT.md)
for the design rationale and verification record.

## Model and data flow

1. `pdfplumber` extracts page text, headers, word boxes, and page dimensions;
   sparse pages can use Tesseract OCR.
2. A fast Hugging Face tokenizer creates sentence-aware overlapping chunks and
   retains global character offsets.
3. Page, section, and normalized two-dimensional box embeddings are fused with
   encoder token states. Each layout source can be disabled.
4. A two-layer document Transformer shares information between chunk states.
5. Joint heads predict exact-span BIO entities, zero or more neutral finding
   categories, and typed entity relations including `NO_RELATION` negatives.
6. Inference JSON links predictions to character spans, sentences, and source
   pages.

Finding prediction is a ten-label, non-exclusive sigmoid task. The canonical
entity type is `FINDING`, and the neutral relation is `ASSOCIATED_WITH`. Legacy
single-label records are migrated to one-hot targets at schema boundaries.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

Tesseract must also be installed on the operating system if OCR fallback is
enabled. Model/tokenizer weights are downloaded from Hugging Face the first
time a preset is used.

## End-to-end workflow

### 1. Extract observation candidates

For a collected-corpus manifest, segment COA report summaries into unlabeled,
page-linked observation/recommendation units:

```powershell
deeplrn-observations `
  --manifest output/pdf/discrepancy_pilot_corpus/corpus_manifest.json `
  --output output/observation_candidates/discrepancy_pilot_observations.jsonl
```

This structural pass preserves the source PDF, LGU, year, source item, and PDF
pages. It intentionally emits `review_status: "unreviewed"` and no finding
label. Review and correct these candidates before creating annotations.

### 2. Create annotations

The preferred unit is one audit observation with its recommendation. Entity
offsets are zero-based, half-open spans within `observation_text`.

```json
{
  "schema_version": 2,
  "doc_id": "sample-lgu-2024-obs-001",
  "parent_doc_id": "sample-lgu-2024",
  "source_pdf": "pdfs/sample.pdf",
  "lgu": "Sample LGU",
  "year": 2024,
  "observation_text": "Example Builders received an unsupported payment.",
  "recommendation_text": "Require complete supporting documents.",
  "evidence_page_numbers": [3],
  "finding_labels": ["procurement_irregularity", "unsupported_disbursement"],
  "entities": [
    {
      "entity_id": "e1",
      "label": "CONTRACTOR",
      "start_char": 0,
      "end_char": 16,
      "text": "Example Builders",
      "page_number": 3
    }
  ],
  "relations": []
}
```

The complete validator is in `deeplrn/schema.py`. Preparing supervised labels
is not the same as conducting a human-review or usability study; this project
does not implement such a study. If no reliable labeled data can be produced,
the software can still run but defensible model-quality claims cannot be made.

Join extracted candidates to adjudicated overlays with:

```bash
python -m deeplrn annotations --candidates candidates.jsonl --overlays reviewed.json --output annotations
```

The command rejects overlays whose `training_eligible` flag is false. For an
explicitly weak-supervised pilot only, `--allow-machine-drafts` preserves that
status as `metadata.supervision_quality: "machine_draft"`; it does not convert
the labels into human gold.

Training and baseline commands reject those weak records by default. A pilot
must additionally pass `--allow-weak-supervision`, making the provenance choice
visible in the command and saved training configuration.

### 3. Prepare model records

```bash
deeplrn-prepare --annotations annotations --pdf-root . --output records --preset deeplrn
```

For observation annotations, this uses embedded text and does not reopen the
PDF. Legacy full-document annotations still use PDF extraction and deterministic
character-to-token alignment.
Each record contains token IDs, BIO labels, finding labels, positive and
negative relation candidates, page/section/box features, sentence IDs, and
global evidence offsets.

### 4. Make leakage-resistant splits

```bash
deeplrn-split --records records --output manifests/split.json --seed 13
```

Every year from one LGU remains in one partition. MinHash candidate search plus
exact shingle Jaccard confirmation also prevents near-duplicate reports from
crossing partitions.

### 5. Train and evaluate

```bash
deeplrn-train --manifest manifests/split.json --output checkpoints --epochs 10 --seed 42 \
  --selection-metric finding_macro_f1
```

Training saves `best.pt`, periodic checkpoints, and `final.pt`. A checkpoint
contains its label order, model and training configurations, optimizer and
scheduler state, completed epochs, global step, and metrics. Resume with:

```bash
deeplrn-train --manifest manifests/split.json --output checkpoints --resume checkpoints/final.pt
```

Evaluate without training:

```bash
deeplrn-train --manifest manifests/split.json --output checkpoints --resume checkpoints/best.pt --eval-only
```

Reported measures are exact entity-span precision/recall/F1, multi-label
finding subset accuracy plus macro/micro F1 and per-label scores, binary Brier
score and expected calibration error, positive-only relation F1, and
evidence-aware tuple F1. `NO_RELATION` true negatives do not inflate relation
F1. Unreviewed NER/relation tasks are masked, so empty draft arrays do not
silently become negative gold labels.

### 6. Run evidence-linked inference

```bash
deeplrn-infer --checkpoint checkpoints/best.pt --input report.pdf --output result.json
```

The output uses a neutral `finding` object and includes entity character spans,
page numbers, sentence IDs, confidence scores, typed relationships, and their
combined evidence pages.

## Baselines and ablations

List the reproducible neural presets:

```bash
deeplrn-experiments list
```

The presets are `deeplrn`, `sentence_roberta`, `longformer`, and `layoutlmv3`.
Each requires records prepared with that preset because token IDs are not
interchangeable between encoder families.

Train the proposal's TF-IDF plus linear-SVM finding baseline:

```bash
deeplrn-experiments tfidf-svm --manifest manifests/split.json --output baselines/tfidf-svm.joblib
```

Examples of controlled ablations:

```bash
# Remove all layout features
deeplrn-train --manifest manifests/split.json --text-only

# Remove only box positions or cross-chunk attention
deeplrn-train --manifest manifests/split.json --no-bbox
deeplrn-train --manifest manifests/split.json --no-document-context

# Remove an auxiliary task from the joint objective
deeplrn-train --manifest manifests/split.json --relation-loss-weight 0
```

`deeplrn/statistics.py` provides seeded LGU-cluster bootstrap intervals, paired
cluster permutation tests, and Holm multiple-comparison correction. These
utilities require actual per-LGU test counts; the repository does not fabricate
research results.

## Verification

```bash
python -m pytest -q
```

Tests are offline and cover extraction helpers, chunking, schemas, annotation
alignment, leakage controls, layout fusion and ablations, metrics, statistical
tests, baselines, manifest selection, checkpoint contracts, and an end-to-end
one-epoch training/resume path.

## Current boundary

The implementation can now prepare, split, train, evaluate, compare, resume,
and infer. To finish the research rather than the software, the project still
needs legally usable COA source documents, a reliable labeled corpus, fixed-seed
training runs for every comparison, and held-out-LGU reporting. No accuracy,
F1, confidence interval, or superiority claim is currently supported.
