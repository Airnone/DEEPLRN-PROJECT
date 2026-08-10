"""Convert extracted pages and human annotations into model-ready JSON."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import json

from deeplrn.config import CHUNK_CFG, LAYOUT_CFG, NER_CFG, RELATION_CFG, ChunkConfig, LayoutConfig
from deeplrn.preprocessing.chunker import (
    DocumentChunker,
    TextChunk,
    _split_sentences,
    concatenate_pages,
)
from deeplrn.preprocessing.pdf_extractor import PageData, WordBox
from deeplrn.schema import AnnotatedDocument, FINDING_LABELS


@dataclass(frozen=True)
class AlignedWord:
    start_char: int
    end_char: int
    page_number: int
    bbox: Tuple[int, int, int, int]


def _normalise_bbox(word: WordBox, page: PageData, bins: int) -> Tuple[int, int, int, int]:
    width = max(float(page.metadata.get("width", 1.0)), 1.0)
    height = max(float(page.metadata.get("height", 1.0)), 1.0)
    raw = (
        round(word.x0 / width * (bins - 1)),
        round(word.top / height * (bins - 1)),
        round(word.x1 / width * (bins - 1)),
        round(word.bottom / height * (bins - 1)),
    )
    return tuple(max(0, min(bins - 1, int(value))) for value in raw)


def _page_lookup(page_map: Sequence[Tuple[int, int, int]], position: int) -> int:
    for start, end, page_number in page_map:
        if start <= position < end:
            return page_number
    return 0


def _align_words(
    pages: Sequence[PageData],
    page_map: Sequence[Tuple[int, int, int]],
    layout_cfg: LayoutConfig,
) -> List[AlignedWord]:
    page_offsets = {page_number: start for start, _, page_number in page_map}
    aligned: List[AlignedWord] = []
    for page in pages:
        global_offset = page_offsets.get(page.page_number)
        if global_offset is None:
            continue
        cursor = 0
        lowered = page.text.lower()
        for word in page.word_boxes:
            local_start = page.text.find(word.text, cursor)
            if local_start < 0:
                local_start = lowered.find(word.text.lower(), cursor)
            if local_start < 0:
                continue
            local_end = local_start + len(word.text)
            aligned.append(
                AlignedWord(
                    start_char=global_offset + local_start,
                    end_char=global_offset + local_end,
                    page_number=page.page_number,
                    bbox=_normalise_bbox(word, page, layout_cfg.bbox_bins),
                )
            )
            cursor = local_end
    return aligned


def _section_starts(
    pages: Sequence[PageData], page_map: Sequence[Tuple[int, int, int]]
) -> List[Tuple[int, int]]:
    page_offsets = {page_number: start for start, _, page_number in page_map}
    starts: List[Tuple[int, int]] = [(0, 0)]
    section_id = 0
    for page in pages:
        page_offset = page_offsets.get(page.page_number)
        if page_offset is None:
            continue
        cursor = 0
        for header in page.headers:
            position = page.text.find(header, cursor)
            if position < 0:
                continue
            section_id += 1
            starts.append((page_offset + position, section_id))
            cursor = position + len(header)
    return starts


def _sentence_spans(document_text: str) -> List[Tuple[int, int, int]]:
    spans: List[Tuple[int, int, int]] = []
    cursor = 0
    for sentence_id, sentence in enumerate(_split_sentences(document_text)):
        start = document_text.find(sentence, cursor)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        spans.append((start, end, sentence_id))
        cursor = end
    return spans


def _interval_value(spans: Sequence[Tuple[int, int, int]], position: int, default: int) -> int:
    for start, end, value in spans:
        if start <= position < end:
            return value
    return default


def _last_section(section_starts: Sequence[Tuple[int, int]], position: int) -> int:
    value = 0
    for start, section_id in section_starts:
        if start > position:
            break
        value = section_id
    return value


class TrainingRecordBuilder:
    """Build a deterministic training record from pages and annotations."""

    def __init__(
        self,
        tokenizer: Any,
        chunk_config: ChunkConfig | None = None,
        layout_config: LayoutConfig | None = None,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("a fast tokenizer is required for offset mapping")
        self.tokenizer = tokenizer
        self.chunk_config = chunk_config or CHUNK_CFG
        self.layout_config = layout_config or LAYOUT_CFG
        self.tag_to_id = {tag: index for index, tag in enumerate(NER_CFG.tags)}
        self.finding_to_id = {label: index for index, label in enumerate(FINDING_LABELS)}
        self.relation_to_id = {
            label: index for index, label in enumerate(RELATION_CFG.relation_types, start=1)
        }

    def build(self, pages: Sequence[PageData], annotation: AnnotatedDocument) -> Dict[str, Any]:
        annotation.validate()
        ner_reviewed = bool(
            annotation.metadata.get("ner_reviewed", bool(annotation.entities))
        )
        relations_reviewed = bool(
            annotation.metadata.get("relations_reviewed", bool(annotation.relations))
        )
        document_text, page_map = concatenate_pages(list(pages))
        for entity in annotation.entities:
            if entity.end_char > len(document_text):
                raise ValueError(
                    f"entity {entity.entity_id} ends beyond document text ({len(document_text)})"
                )
            if entity.text and document_text[entity.start_char:entity.end_char] != entity.text:
                raise ValueError(f"entity text mismatch for {entity.entity_id}")

        chunks = DocumentChunker(
            config=self.chunk_config, tokenizer=self.tokenizer
        ).chunk_pages(list(pages))
        words = _align_words(pages, page_map, self.layout_config)
        sections = _section_starts(pages, page_map)
        sentences = _sentence_spans(document_text)

        serialized_chunks: List[Dict[str, Any]] = []
        entity_locations: Dict[str, List[Tuple[int, int, int, int]]] = {
            entity.entity_id: [] for entity in annotation.entities
        }
        for chunk in chunks:
            item, locations = self._build_chunk(
                chunk, annotation.entities, page_map, words, sections, sentences
            )
            if not ner_reviewed:
                item["ner_labels"] = [-100] * len(item["ner_labels"])
            serialized_chunks.append(item)
            for entity_id, location in locations.items():
                entity_locations[entity_id].append(location)

        missing = [entity_id for entity_id, locations in entity_locations.items() if not locations]
        if missing:
            raise ValueError(f"entities could not be aligned to tokens: {missing}")
        selected = {
            entity_id: self._select_location(locations)
            for entity_id, locations in entity_locations.items()
        }

        if relations_reviewed:
            relation_triples, relation_candidate_metadata = self._relation_candidates(
                annotation, selected
            )
        else:
            relation_triples, relation_candidate_metadata = [], []
        finding_label_ids = sorted(
            self.finding_to_id[label] for label in annotation.finding_labels
        )
        record = {
            "schema_version": annotation.schema_version,
            "doc_id": annotation.doc_id,
            "parent_doc_id": annotation.parent_doc_id or annotation.doc_id,
            "source_pdf": annotation.source_pdf,
            "lgu": annotation.lgu,
            "year": annotation.year,
            "document_sha256": sha256(document_text.encode("utf-8")).hexdigest(),
            "document_text": document_text,
            "observation_text": annotation.observation_text,
            "recommendation_text": annotation.recommendation_text,
            "evidence_page_numbers": list(annotation.evidence_page_numbers),
            "finding_labels": list(annotation.finding_labels),
            "finding_label_ids": finding_label_ids,
            "finding_label_vector": [
                int(index in finding_label_ids) for index in range(len(FINDING_LABELS))
            ],
            "task_annotations": {
                "finding": True,
                "ner": ner_reviewed,
                "relations": relations_reviewed,
            },
            "annotation_metadata": dict(annotation.metadata),
            "pad_token_id": int(getattr(self.tokenizer, "pad_token_id", 1) or 1),
            "chunks": serialized_chunks,
            "relation_triples": relation_triples,
            "relation_candidate_metadata": relation_candidate_metadata,
            "entities": [
                {
                    "entity_id": entity.entity_id,
                    "label": entity.label,
                    "start_char": entity.start_char,
                    "end_char": entity.end_char,
                    "page_number": entity.page_number,
                }
                for entity in annotation.entities
            ],
            "relations": [
                {
                    "relation_id": relation.relation_id,
                    "head_entity_id": relation.head_entity_id,
                    "tail_entity_id": relation.tail_entity_id,
                    "label": relation.label,
                    "evidence_page_numbers": list(relation.evidence_page_numbers),
                }
                for relation in annotation.relations
            ],
        }
        if len(finding_label_ids) == 1:
            # Retained for readers of version-1 prepared records.
            record["finding_label"] = annotation.finding_labels[0]
            record["finding_label_id"] = finding_label_ids[0]
        return record

    def _build_chunk(
        self,
        chunk: TextChunk,
        entities: Sequence[Any],
        page_map: Sequence[Tuple[int, int, int]],
        words: Sequence[AlignedWord],
        sections: Sequence[Tuple[int, int]],
        sentences: Sequence[Tuple[int, int, int]],
    ) -> Tuple[Dict[str, Any], Dict[str, Tuple[int, int, int, int]]]:
        encoded = self.tokenizer(
            chunk.text,
            max_length=self.chunk_config.max_tokens,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        special = list(encoded["special_tokens_mask"])

        global_offsets: List[List[int]] = []
        page_ids: List[int] = []
        section_ids: List[int] = []
        bboxes: List[List[int]] = []
        sentence_ids: List[int] = []
        ner_labels = [-100] * len(input_ids)

        for (local_start, local_end), is_special, attended in zip(
            offsets, special, attention_mask
        ):
            if is_special or not attended or local_end <= local_start:
                global_offsets.append([-1, -1])
                page_ids.append(0)
                section_ids.append(0)
                bboxes.append([0, 0, 0, 0])
                sentence_ids.append(-1)
                continue

            start = chunk.char_offset_start + local_start
            end = chunk.char_offset_start + local_end
            midpoint = start + max(0, end - start - 1) // 2
            global_offsets.append([start, end])
            page_ids.append(_page_lookup(page_map, midpoint))
            section_ids.append(_last_section(sections, midpoint))
            sentence_ids.append(_interval_value(sentences, midpoint, -1))
            word = next(
                (item for item in words if item.start_char < end and item.end_char > start),
                None,
            )
            bboxes.append(list(word.bbox) if word else [0, 0, 0, 0])
            ner_labels[len(global_offsets) - 1] = self.tag_to_id["O"]

        locations: Dict[str, Tuple[int, int, int, int]] = {}
        for entity in entities:
            indexes = [
                index
                for index, (start, end) in enumerate(global_offsets)
                if start >= 0 and start < entity.end_char and end > entity.start_char
            ]
            if not indexes:
                continue
            first, last = indexes[0], indexes[-1] + 1
            ner_labels[first] = self.tag_to_id[f"B-{entity.label}"]
            for index in indexes[1:]:
                ner_labels[index] = self.tag_to_id[f"I-{entity.label}"]
            sentence_id = next(
                (sentence_ids[index] for index in indexes if sentence_ids[index] >= 0), -1
            )
            locations[entity.entity_id] = (chunk.chunk_id, first, last, sentence_id)

        return (
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "page_numbers": chunk.page_numbers,
                "char_offset_start": chunk.char_offset_start,
                "char_offset_end": chunk.char_offset_end,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "ner_labels": ner_labels,
                "page_ids": page_ids,
                "section_ids": section_ids,
                "bboxes": bboxes,
                "sentence_ids": sentence_ids,
                "token_offsets": global_offsets,
            },
            locations,
        )

    @staticmethod
    def _select_location(
        locations: Sequence[Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int]:
        return sorted(locations, key=lambda item: (-(item[2] - item[1]), item[0]))[0]

    def _relation_candidates(
        self,
        annotation: AnnotatedDocument,
        locations: Mapping[str, Tuple[int, int, int, int]],
    ) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
        positives = {
            (relation.head_entity_id, relation.tail_entity_id): relation
            for relation in annotation.relations
        }
        entity_pages = {
            entity.entity_id: entity.page_number
            for entity in annotation.entities
            if entity.page_number is not None
        }
        triples: List[List[int]] = []
        candidate_metadata: List[Dict[str, Any]] = []
        entity_ids = sorted(locations)
        for head_id in entity_ids:
            for tail_id in entity_ids:
                if head_id == tail_id:
                    continue
                hc, hs, he, head_sentence = locations[head_id]
                tc, ts, te, tail_sentence = locations[tail_id]
                if head_sentence >= 0 and tail_sentence >= 0:
                    if abs(head_sentence - tail_sentence) > RELATION_CFG.sentence_window:
                        continue
                relation = positives.get((head_id, tail_id))
                label_id = self.relation_to_id[relation.label] if relation else 0
                triples.append([hc, hs, he, tc, ts, te, label_id])
                evidence_pages = list(relation.evidence_page_numbers) if relation else []
                if relation and not evidence_pages:
                    evidence_pages = sorted(
                        {
                            page
                            for page in (entity_pages.get(head_id), entity_pages.get(tail_id))
                            if page is not None
                        }
                    )
                candidate_metadata.append(
                    {
                        "head_entity_id": head_id,
                        "tail_entity_id": tail_id,
                        "gold_label_id": label_id,
                        "evidence_page_numbers": evidence_pages,
                    }
                )
        return triples, candidate_metadata


def save_training_record(record: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(dict(record), stream, indent=2, ensure_ascii=False)


def build_inference_chunks(
    pages: Sequence[PageData],
    chunks: Sequence[TextChunk],
    tokenizer: Any,
    chunk_config: ChunkConfig | None = None,
    layout_config: LayoutConfig | None = None,
) -> List[Dict[str, Any]]:
    """Build the same token/layout features used during training, without labels."""
    builder = TrainingRecordBuilder(tokenizer, chunk_config, layout_config)
    document_text, page_map = concatenate_pages(list(pages))
    words = _align_words(pages, page_map, builder.layout_config)
    sections = _section_starts(pages, page_map)
    sentences = _sentence_spans(document_text)
    result = []
    for chunk in chunks:
        item, _ = builder._build_chunk(
            chunk,
            (),
            page_map,
            words,
            sections,
            sentences,
        )
        item.pop("ner_labels", None)
        result.append(item)
    return result
