"""Model-ready DEEPLRN dataset and document collation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import random

import torch
from torch.utils.data import Dataset


RelationTriple = Tuple[int, int, int, int, int, int, int]


@dataclass
class DocumentSample:
    doc_id: str
    chunk_input_ids: torch.Tensor
    chunk_attention_masks: torch.Tensor
    ner_labels: torch.Tensor
    finding_label: int
    relation_triples: List[RelationTriple]
    num_chunks: int
    page_ids: torch.Tensor
    section_ids: torch.Tensor
    bboxes: torch.Tensor
    sentence_ids: torch.Tensor
    token_offsets: torch.Tensor
    metadata: Dict[str, Any]

    @property
    def violation_label(self) -> int:
        """Legacy alias retained while callers migrate to finding terminology."""
        return self.finding_label


def _pad_list(values: list, length: int, pad_value: Any) -> list:
    values = list(values[:length])
    if len(values) < length:
        values.extend([pad_value for _ in range(length - len(values))])
    return values


class DeepLRNDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        max_tokens: int = 384,
        *,
        files: List[str | Path] | None = None,
    ):
        self.data_dir = Path(data_dir) if data_dir is not None else Path(".")
        self.max_tokens = max_tokens
        if files is not None:
            self.doc_files = sorted(Path(path) for path in files)
        else:
            self.doc_files = (
                sorted(self.data_dir.glob("*.json")) if self.data_dir.exists() else []
            )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        split: str,
        max_tokens: int = 384,
    ) -> "DeepLRNDataset":
        """Load exactly one named split from a leakage-control manifest."""

        manifest_path = Path(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        files = []
        for document in manifest.get("documents", []):
            if document.get("split") != split:
                continue
            source = Path(document["source_path"])
            if not source.is_absolute():
                source = manifest_path.parent / source
            files.append(source)
        return cls(max_tokens=max_tokens, files=files)

    def __len__(self) -> int:
        return len(self.doc_files)

    def __getitem__(self, idx: int) -> DocumentSample:
        path = self.doc_files[idx]
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)

        doc_id = str(data.get("doc_id", path.stem))
        chunks = data.get("chunks", [])
        pad_token_id = int(data.get("pad_token_id", 1))

        input_rows: List[list] = []
        mask_rows: List[list] = []
        ner_rows: List[list] = []
        page_rows: List[list] = []
        section_rows: List[list] = []
        bbox_rows: List[list] = []
        sentence_rows: List[list] = []
        offset_rows: List[list] = []

        for chunk in chunks:
            raw_ids = list(chunk.get("input_ids", []))
            raw_length = len(raw_ids)
            input_rows.append(_pad_list(raw_ids, self.max_tokens, pad_token_id))
            mask_rows.append(_pad_list(chunk.get("attention_mask", []), self.max_tokens, 0))
            ner_rows.append(_pad_list(chunk.get("ner_labels", []), self.max_tokens, -100))
            page_rows.append(_pad_list(chunk.get("page_ids", [0] * raw_length), self.max_tokens, 0))
            section_rows.append(
                _pad_list(chunk.get("section_ids", [0] * raw_length), self.max_tokens, 0)
            )
            bbox_rows.append(
                _pad_list(
                    chunk.get("bboxes", [[0, 0, 0, 0] for _ in range(raw_length)]),
                    self.max_tokens,
                    [0, 0, 0, 0],
                )
            )
            sentence_rows.append(
                _pad_list(chunk.get("sentence_ids", [-1] * raw_length), self.max_tokens, -1)
            )
            offset_rows.append(
                _pad_list(
                    chunk.get("token_offsets", [[-1, -1] for _ in range(raw_length)]),
                    self.max_tokens,
                    [-1, -1],
                )
            )

        if not input_rows:
            input_rows = [[pad_token_id] * self.max_tokens]
            mask_rows = [[0] * self.max_tokens]
            ner_rows = [[-100] * self.max_tokens]
            page_rows = [[0] * self.max_tokens]
            section_rows = [[0] * self.max_tokens]
            bbox_rows = [[[0, 0, 0, 0] for _ in range(self.max_tokens)]]
            sentence_rows = [[-1] * self.max_tokens]
            offset_rows = [[[-1, -1] for _ in range(self.max_tokens)]]

        finding_label = int(data.get("finding_label_id", data.get("violation_label", 0)))
        return DocumentSample(
            doc_id=doc_id,
            chunk_input_ids=torch.tensor(input_rows, dtype=torch.long),
            chunk_attention_masks=torch.tensor(mask_rows, dtype=torch.long),
            ner_labels=torch.tensor(ner_rows, dtype=torch.long),
            finding_label=finding_label,
            relation_triples=[tuple(item) for item in data.get("relation_triples", [])],
            num_chunks=len(input_rows),
            page_ids=torch.tensor(page_rows, dtype=torch.long),
            section_ids=torch.tensor(section_rows, dtype=torch.long),
            bboxes=torch.tensor(bbox_rows, dtype=torch.long),
            sentence_ids=torch.tensor(sentence_rows, dtype=torch.long),
            token_offsets=torch.tensor(offset_rows, dtype=torch.long),
            metadata={
                "doc_id": doc_id,
                "lgu": data.get("lgu", ""),
                "year": data.get("year"),
                "source_pdf": data.get("source_pdf", ""),
                "entities": data.get("entities", []),
                "relations": data.get("relations", []),
                "relation_candidate_metadata": data.get(
                    "relation_candidate_metadata", []
                ),
            },
        )


def collate_documents(batch: List[DocumentSample]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")

    batch_size = len(batch)
    max_chunks = max(sample.num_chunks for sample in batch)
    max_tokens = batch[0].chunk_input_ids.size(1)

    input_ids = torch.ones((batch_size, max_chunks, max_tokens), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_chunks, max_tokens), dtype=torch.long)
    chunk_mask = torch.zeros((batch_size, max_chunks), dtype=torch.bool)
    ner_labels = torch.full((batch_size, max_chunks, max_tokens), -100, dtype=torch.long)
    finding_labels = torch.zeros(batch_size, dtype=torch.long)
    page_ids = torch.zeros((batch_size, max_chunks, max_tokens), dtype=torch.long)
    section_ids = torch.zeros((batch_size, max_chunks, max_tokens), dtype=torch.long)
    bboxes = torch.zeros((batch_size, max_chunks, max_tokens, 4), dtype=torch.long)
    sentence_ids = torch.full((batch_size, max_chunks, max_tokens), -1, dtype=torch.long)
    token_offsets = torch.full((batch_size, max_chunks, max_tokens, 2), -1, dtype=torch.long)

    relation_triples = []
    doc_ids = []
    metadata = []
    for index, sample in enumerate(batch):
        count = sample.num_chunks
        input_ids[index, :count] = sample.chunk_input_ids
        attention_mask[index, :count] = sample.chunk_attention_masks
        chunk_mask[index, :count] = True
        ner_labels[index, :count] = sample.ner_labels
        page_ids[index, :count] = sample.page_ids
        section_ids[index, :count] = sample.section_ids
        bboxes[index, :count] = sample.bboxes
        sentence_ids[index, :count] = sample.sentence_ids
        token_offsets[index, :count] = sample.token_offsets
        finding_labels[index] = sample.finding_label
        relation_triples.append(sample.relation_triples)
        doc_ids.append(sample.doc_id)
        metadata.append(sample.metadata)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "chunk_mask": chunk_mask,
        "ner_labels": ner_labels,
        "finding_labels": finding_labels,
        "violation_labels": finding_labels,  # temporary legacy alias
        "relation_triples": relation_triples,
        "page_ids": page_ids,
        "section_ids": section_ids,
        "bboxes": bboxes,
        "sentence_ids": sentence_ids,
        "token_offsets": token_offsets,
        "doc_ids": doc_ids,
        "metadata": metadata,
    }


def create_dummy_dataset(output_dir: str | Path, num_docs: int = 5) -> None:
    """Create structural smoke-test data; never use it for model evaluation."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for index in range(num_docs):
        chunk_count = random.randint(1, 4)
        chunks = []
        for _ in range(chunk_count):
            length = random.randint(10, 50)
            chunks.append(
                {
                    "input_ids": [random.randint(2, 1000) for _ in range(length)],
                    "attention_mask": [1] * length,
                    "ner_labels": [random.choice([0, 1, 2]) for _ in range(length)],
                    "page_ids": [1] * length,
                    "section_ids": [0] * length,
                    "bboxes": [[0, 0, 0, 0] for _ in range(length)],
                    "sentence_ids": [0] * length,
                    "token_offsets": [[position, position + 1] for position in range(length)],
                }
            )
        record = {
            "doc_id": f"dummy_doc_{index}",
            "chunks": chunks,
            "finding_label_id": random.randint(0, 4),
            "relation_triples": [],
        }
        with (target / f"doc_{index}.json").open("w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
