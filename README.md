# DEEPLRN — Document-Level NLP for Philippine COA Audit Reports

A multi-task NLP pipeline that extracts financial violation details—entities, violation types, and relationships—from long, unstructured Philippine Commission on Audit (COA) PDF reports.

## Architecture Overview

```
PDF Report
    │
    ▼
┌─────────────────────────────┐
│  PDF Extractor              │  pdfplumber + Tesseract OCR fallback
│  (page text, bounding boxes,│
│   headers)                  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Document Chunker           │  384-token windows, 64-token overlap
│  (sentence-boundary aware)  │  RoBERTa BPE tokenisation
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  RoBERTa Encoder            │  Per-chunk contextual embeddings
│  + Document Transformer     │  Cross-chunk attention (2 layers)
└──────────────┬──────────────┘
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
   ┌──────┐ ┌──────┐ ┌──────────┐
   │ NER  │ │Class.│ │ Relation │   Multi-task heads
   │ Head │ │ Head │ │ Extract. │   (trained jointly)
   └──┬───┘ └──┬───┘ └────┬─────┘
      │        │           │
      ▼        ▼           ▼
   Entities  Violation   Entity
             Category    Links
               │
               ▼
        JSON Output
```

## Quick Start

### 1. Install dependencies

```bash
pip install pdfplumber pytesseract Pillow transformers torch tqdm
```

> **Note:** For OCR fallback you also need the [Tesseract binary](https://github.com/tesseract-ocr/tesseract) installed on your system.

### 2. Preprocess PDFs

```bash
# Single PDF
python -m deeplrn --input report.pdf --output output/

# Directory of PDFs
python -m deeplrn --input pdfs/ --output output/
```

### 3. Train the model

```python
from deeplrn.model import DeepLRNModel, ModelConfig
from deeplrn.training import DeepLRNDataset, Trainer, TrainingConfig, create_dummy_dataset

# Generate dummy data for testing
create_dummy_dataset("data/train", num_docs=20)
create_dummy_dataset("data/eval", num_docs=5)

# Load data
train_ds = DeepLRNDataset("data/train")
eval_ds = DeepLRNDataset("data/eval")

# Create model
model = DeepLRNModel(ModelConfig(freeze_layers=6))

# Train
trainer = Trainer(model, train_ds, eval_ds, TrainingConfig(num_epochs=5))
trainer.train()
```

### 4. Run inference

```python
from deeplrn.pipeline import InferencePipeline

pipe = InferencePipeline.from_checkpoint("checkpoints/best_model.pt")
result = pipe.run("report.pdf")
print(result["violation"])      # {'type': 'procurement_irregularity', 'confidence': 0.92}
print(result["entities"][:3])   # [{'text': 'ABC Construction', 'type': 'CONTRACTOR'}, ...]
print(result["relationships"])  # [{'head': ..., 'tail': ..., 'relation': 'RESPONSIBLE_FOR'}]
```

### 5. JSON Output Format

```json
{
  "document": {
    "filename": "report.pdf",
    "total_pages": 45,
    "total_chunks": 128,
    "inference_time_s": 12.3
  },
  "violation": {
    "predicted_type": "procurement_irregularity",
    "confidence": 0.92,
    "all_scores": { ... }
  },
  "entities": [
    {"text": "ABC Construction", "type": "CONTRACTOR", "confidence": 0.95, ...},
    {"text": "PHP 2,500,000", "type": "AMOUNT", "confidence": 0.98, ...}
  ],
  "relationships": [
    {"head": {"text": "ABC Construction"}, "tail": {"text": "procurement irregularity"},
     "relation": "INVOLVES", "confidence": 0.87}
  ]
}
```

## Project Structure

```
DEEPLRN-PROJECT/
├── deeplrn/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                          # CLI for preprocessing
│   ├── config.py                       # Central configuration
│   ├── pipeline.py                     # End-to-end inference
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py            # PDF text extraction + OCR
│   │   └── chunker.py                  # Overlapping-window chunking
│   ├── model/
│   │   ├── __init__.py
│   │   ├── encoder.py                  # RoBERTa chunk encoder
│   │   ├── doc_transformer.py          # 2-layer document Transformer
│   │   ├── deeplrn_model.py            # Unified multi-task model
│   │   └── heads/
│   │       ├── __init__.py
│   │       ├── ner.py                  # NER sequence labeling
│   │       ├── classifier.py           # Violation classification
│   │       └── relation.py             # Relation extraction (bilinear)
│   └── training/
│       ├── __init__.py
│       ├── dataset.py                  # PyTorch dataset + collation
│       └── trainer.py                  # Multi-task training loop
├── tests/
│   ├── test_chunker.py
│   └── test_integration.py
├── pyproject.toml
└── README.md
```

## Pipeline Stages

| Stage | Status | Module |
|-------|--------|--------|
| PDF Extraction + OCR | ✅ Done | `preprocessing.pdf_extractor` |
| Token Chunking (384/64) | ✅ Done | `preprocessing.chunker` |
| RoBERTa Chunk Encoder | ✅ Done | `model.encoder` |
| Document Transformer (2-layer) | ✅ Done | `model.doc_transformer` |
| NER Head (17 BIO tags) | ✅ Done | `model.heads.ner` |
| Violation Classifier (5 classes) | ✅ Done | `model.heads.classifier` |
| Relation Extractor (bilinear) | ✅ Done | `model.heads.relation` |
| Unified Multi-Task Model | ✅ Done | `model.deeplrn_model` |
| Training Loop (AdamW + warmup) | ✅ Done | `training.trainer` |
| Dataset + Collation | ✅ Done | `training.dataset` |
| Inference Pipeline (PDF → JSON) | ✅ Done | `pipeline` |

## NER Tag Schema

`PERSON`, `ORGANIZATION`, `LGU`, `CONTRACTOR`, `AMOUNT`, `DATE`, `PROJECT`, `VIOLATION` — using BIO encoding (17 tags).

## Violation Categories

1. Unauthorized Expenditure
2. Unliquidated Cash Advance
3. Procurement Irregularity
4. Unsupported Disbursement
5. Suspicious Contractor Activity

## Relation Types

- `INVOLVES` — links a violation to an entity
- `AMOUNT_OF` — links a monetary amount to a violation
- `RESPONSIBLE_FOR` — links a person/org to a violation