# DEEPLRN

DEEPLRN is an evidence-linked, document-level NLP system for Philippine
Commission on Audit reports. It extracts neutral audit findings, entities,
amounts, projects, and textual relationships. Its output is a review aid, not a
determination of misconduct, intent, liability, or guilt.

The repository is **software-complete and training-ready**. It does not contain
the proposed COA corpus, human labels, trained research checkpoint, or measured
held-out results. Those data-dependent deliverables must not be inferred from
the passing software tests. See [DEVELOPMENT.md](DEVELOPMENT.md) for the design
rationale and verification record.

## Model and data flow

1. `pdfplumber` extracts page text, headers, word boxes, and page dimensions;
   sparse pages can use Tesseract OCR.
2. A fast Hugging Face tokenizer creates sentence-aware overlapping chunks and
   retains global character offsets.
3. Page, section, and normalized two-dimensional box embeddings are fused with
   encoder token states. Each layout source can be disabled.
4. A two-layer document Transformer shares information between chunk states.
5. Joint heads predict exact-span BIO entities, one neutral finding category,
   and typed entity relations including `NO_RELATION` negatives.
6. Inference JSON links predictions to character spans, sentences, and source
   pages.

The canonical entity type is `FINDING`, the fifth finding class is
`contractor_related_concern`, and the neutral relation is `ASSOCIATED_WITH`.
Legacy prototype labels are accepted only at schema-conversion boundaries.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
```

Tesseract must also be installed on the operating system if OCR fallback is
enabled. Model/tokenizer weights are downloaded from Hugging Face the first
time a preset is used.

## End-to-end workflow

### 1. Create annotations

Offsets use the exact document text obtained by joining extracted pages with a
newline. They are zero-based, half-open character spans.

```json
{
  "schema_version": 1,
  "doc_id": "sample-lgu-2024",
  "source_pdf": "pdfs/sample.pdf",
  "lgu": "Sample LGU",
  "year": 2024,
  "finding_label": "procurement_irregularity",
  "entities": [
    {
      "entity_id": "e1",
      "label": "CONTRACTOR",
      "start_char": 120,
      "end_char": 136,
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

### 2. Prepare model records

```bash
deeplrn-prepare --annotations annotations --pdf-root . --output records --preset deeplrn
```

This performs PDF extraction and deterministic character-to-token alignment.
Each record contains token IDs, BIO labels, finding labels, positive and
negative relation candidates, page/section/box features, sentence IDs, and
global evidence offsets.

### 3. Make leakage-resistant splits

```bash
deeplrn-split --records records --output manifests/split.json --seed 13
```

Every year from one LGU remains in one partition. MinHash candidate search plus
exact shingle Jaccard confirmation also prevents near-duplicate reports from
crossing partitions.

### 4. Train and evaluate

```bash
deeplrn-train --manifest manifests/split.json --output checkpoints --epochs 10 --seed 42
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

Reported measures are exact entity-span precision/recall/F1, finding accuracy
and macro F1, finding Brier score and expected calibration error, positive-only
relation F1, and evidence-aware tuple F1. `NO_RELATION` true negatives do not
inflate relation F1.

### 5. Run evidence-linked inference

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
