"""Evidence-linked PDF-to-JSON inference for a trained DEEPLRN checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import logging
import time

import torch
from transformers import AutoTokenizer

from deeplrn.checkpoints import load_checkpoint
from deeplrn.config import (
    CHUNK_CFG,
    EXTRACT_CFG,
    FINDING_CFG,
    NER_CFG,
    RELATION_CFG,
    ChunkConfig,
    ExtractionConfig,
)
from deeplrn.preprocessing.chunker import DocumentChunker, TextChunk
from deeplrn.preprocessing.extract import extract_document
from deeplrn.preprocessing.pdf_extractor import PageData
from deeplrn.training.builder import build_inference_chunks


logger = logging.getLogger(__name__)


@dataclass
class Entity:
    text: str
    entity_type: str
    chunk_id: int
    token_start: int
    token_end: int
    page_numbers: List[int] = field(default_factory=list)
    sentence_id: int = -1
    char_start: int = -1
    char_end: int = -1
    confidence: float = 0.0


@dataclass
class Relation:
    head_entity: Entity
    tail_entity: Entity
    relation_type: str
    confidence: float = 0.0


class InferencePipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str = "cpu",
        extract_cfg: ExtractionConfig | None = None,
        chunk_cfg: ChunkConfig | None = None,
        relation_threshold: float = 0.5,
        finding_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= relation_threshold <= 1.0:
            raise ValueError("relation_threshold must be between zero and one")
        if not 0.0 <= finding_threshold <= 1.0:
            raise ValueError("finding_threshold must be between zero and one")
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.extract_cfg = extract_cfg or EXTRACT_CFG
        self.chunk_cfg = chunk_cfg or CHUNK_CFG
        self.relation_threshold = relation_threshold
        self.finding_threshold = finding_threshold
        self.id2tag = {index: tag for index, tag in enumerate(NER_CFG.tags)}
        self.id2finding = {index: label for index, label in enumerate(FINDING_CFG.labels)}
        self.id2relation = {0: "NO_RELATION"}
        self.id2relation.update(
            {index: label for index, label in enumerate(RELATION_CFG.relation_types, start=1)}
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
        relation_threshold: float = 0.5,
        finding_threshold: float = 0.5,
    ) -> "InferencePipeline":
        from deeplrn.model.deeplrn_model import DeepLRNModel, ModelConfig

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model_config = ModelConfig(**checkpoint["model_config"])
        model = DeepLRNModel(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        tokenizer = AutoTokenizer.from_pretrained(model_config.encoder_name, use_fast=True)
        return cls(
            model,
            tokenizer,
            device=device,
            relation_threshold=relation_threshold,
            finding_threshold=finding_threshold,
        )

    @torch.no_grad()
    def run(self, pdf_path: str | Path) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        started = time.perf_counter()
        pages = extract_document(pdf_path, self.extract_cfg)
        chunks = DocumentChunker(
            config=self.chunk_cfg, tokenizer=self.tokenizer
        ).chunk_pages(pages)
        if not chunks:
            return self._empty_result(pdf_path)

        features = self._prepare_inputs(pages, chunks)
        outputs = self.model(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            page_ids=features["page_ids"],
            section_ids=features["section_ids"],
            bboxes=features["bboxes"],
        )
        entities = self._decode_ner(outputs["ner_logits"], features)
        finding = self._decode_finding(outputs["cls_logits"])
        relations = self._predict_relations(entities, features)

        elapsed = time.perf_counter() - started
        result = {
            "document": {
                "filename": pdf_path.name,
                "total_pages": len(pages),
                "total_chunks": len(chunks),
                "inference_time_s": round(elapsed, 2),
            },
            "finding": finding,
            "entities": [self._entity_to_dict(entity) for entity in entities],
            "relationships": [self._relation_to_dict(relation) for relation in relations],
        }
        logger.info(
            "Inference on %s: %d entities, finding=%s (%.3f), %d relations in %.1fs",
            pdf_path.name,
            len(entities),
            finding["predicted_types"],
            finding["confidence"],
            len(relations),
            elapsed,
        )
        return result

    def run_and_save(
        self,
        pdf_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        source = Path(pdf_path)
        result = self.run(source)
        target = Path(output_path) if output_path is not None else source.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
        return result

    def _prepare_inputs(
        self,
        pages: List[PageData],
        chunks: List[TextChunk],
    ) -> Dict[str, torch.Tensor]:
        records = build_inference_chunks(
            pages,
            chunks,
            self.tokenizer,
            chunk_config=self.chunk_cfg,
        )
        fields = (
            "input_ids",
            "attention_mask",
            "page_ids",
            "section_ids",
            "bboxes",
            "sentence_ids",
            "token_offsets",
        )
        return {
            name: torch.tensor([record[name] for record in records], dtype=torch.long)
            .unsqueeze(0)
            .to(self.device)
            for name in fields
        }

    def _decode_ner(
        self,
        ner_logits: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> List[Entity]:
        logits = ner_logits[0]
        predictions = logits.argmax(dim=-1)
        confidences = torch.softmax(logits, dim=-1).max(dim=-1).values
        attention_mask = features["attention_mask"][0]
        entities: List[Entity] = []

        for chunk_index in range(logits.shape[0]):
            tags = predictions[chunk_index].detach().cpu().tolist()
            confidence = confidences[chunk_index].detach().cpu().tolist()
            mask = attention_mask[chunk_index].detach().cpu().tolist()
            current: Dict[str, Any] | None = None

            def close_current() -> None:
                nonlocal current
                if current is None:
                    return
                entities.append(self._build_entity(current, chunk_index, features))
                current = None

            for token_index, tag_id in enumerate(tags):
                if not mask[token_index]:
                    close_current()
                    continue
                tag = self.id2tag.get(tag_id, "O")
                if tag.startswith("B-"):
                    close_current()
                    current = {
                        "type": tag[2:],
                        "start": token_index,
                        "end": token_index + 1,
                        "confidences": [confidence[token_index]],
                    }
                elif tag.startswith("I-"):
                    entity_type = tag[2:]
                    if current is None or current["type"] != entity_type:
                        close_current()
                        current = {
                            "type": entity_type,
                            "start": token_index,
                            "end": token_index + 1,
                            "confidences": [confidence[token_index]],
                        }
                    else:
                        current["end"] = token_index + 1
                        current["confidences"].append(confidence[token_index])
                else:
                    close_current()
            close_current()

        unique: Dict[Tuple[str, int, int, str], Entity] = {}
        for entity in entities:
            key = (entity.entity_type, entity.char_start, entity.char_end, entity.text)
            if key not in unique or entity.confidence > unique[key].confidence:
                unique[key] = entity
        return sorted(unique.values(), key=lambda item: (item.char_start, item.char_end))

    def _build_entity(
        self,
        raw: Dict[str, Any],
        chunk_index: int,
        features: Dict[str, torch.Tensor],
    ) -> Entity:
        start, end = raw["start"], raw["end"]
        token_ids = features["input_ids"][0, chunk_index, start:end].detach().cpu().tolist()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        offsets = (
            features["token_offsets"][0, chunk_index, start:end].detach().cpu().tolist()
        )
        valid_offsets = [pair for pair in offsets if pair[0] >= 0]
        pages = sorted(
            {
                int(value)
                for value in features["page_ids"][0, chunk_index, start:end]
                .detach()
                .cpu()
                .tolist()
                if value > 0
            }
        )
        sentences = [
            int(value)
            for value in features["sentence_ids"][0, chunk_index, start:end]
            .detach()
            .cpu()
            .tolist()
            if value >= 0
        ]
        return Entity(
            text=text,
            entity_type=raw["type"],
            chunk_id=chunk_index,
            token_start=start,
            token_end=end,
            page_numbers=pages,
            sentence_id=sentences[0] if sentences else -1,
            char_start=min(pair[0] for pair in valid_offsets) if valid_offsets else -1,
            char_end=max(pair[1] for pair in valid_offsets) if valid_offsets else -1,
            confidence=sum(raw["confidences"]) / len(raw["confidences"]),
        )

    def _decode_finding(self, logits: torch.Tensor) -> Dict[str, Any]:
        probabilities = torch.sigmoid(logits[0])
        selected = [
            index
            for index, probability in enumerate(probabilities)
            if float(probability.item()) > self.finding_threshold
        ]
        predictions = [
            {
                "type": self.id2finding.get(index, f"unknown_{index}"),
                "confidence": round(float(probabilities[index].item()), 4),
            }
            for index in selected
        ]
        primary_index = max(selected, key=lambda index: float(probabilities[index])) if selected else None
        return {
            "predicted_types": [item["type"] for item in predictions],
            "predictions": predictions,
            "threshold": self.finding_threshold,
            "predicted_type": (
                self.id2finding.get(primary_index, f"unknown_{primary_index}")
                if primary_index is not None
                else None
            ),
            "confidence": (
                round(float(probabilities[primary_index].item()), 4)
                if primary_index is not None
                else 0.0
            ),
            "all_scores": {
                self.id2finding.get(i, f"class_{i}"): round(float(value.item()), 4)
                for i, value in enumerate(probabilities)
            },
        }

    def _predict_relations(
        self,
        entities: List[Entity],
        features: Dict[str, torch.Tensor],
    ) -> List[Relation]:
        if len(entities) < 2:
            return []
        candidate_pairs = []
        for head_index, head in enumerate(entities):
            for tail_index, tail in enumerate(entities):
                if head_index == tail_index:
                    continue
                if head.sentence_id >= 0 and tail.sentence_id >= 0:
                    if abs(head.sentence_id - tail.sentence_id) > RELATION_CFG.sentence_window:
                        continue
                candidate_pairs.append((head_index, tail_index))
        if not candidate_pairs:
            return []

        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]
        sequence_length = input_ids.shape[-1]
        flat_ids = input_ids.view(-1, sequence_length)
        flat_mask = attention_mask.view(-1, sequence_length)
        if getattr(self.model.config, "encoder_uses_2d_positions", False):
            token_embeddings, _ = self.model.encoder(
                flat_ids,
                flat_mask,
                bboxes=features["bboxes"].view(-1, sequence_length, 4),
            )
        else:
            token_embeddings, _ = self.model.encoder(flat_ids, flat_mask)
        if getattr(self.model.config, "use_layout", False):
            token_embeddings = self.model.layout_encoder(
                token_embeddings,
                page_ids=features["page_ids"].view(-1, sequence_length),
                section_ids=features["section_ids"].view(-1, sequence_length),
                bboxes=features["bboxes"].view(-1, sequence_length, 4),
            )

        head_vectors = []
        tail_vectors = []
        for head_index, tail_index in candidate_pairs:
            head = entities[head_index]
            tail = entities[tail_index]
            head_vectors.append(
                token_embeddings[
                    head.chunk_id, head.token_start:head.token_end
                ].mean(dim=0)
            )
            tail_vectors.append(
                token_embeddings[
                    tail.chunk_id, tail.token_start:tail.token_end
                ].mean(dim=0)
            )
        probabilities = torch.softmax(
            self.model.relation_extractor(
                torch.stack(head_vectors), torch.stack(tail_vectors)
            ),
            dim=-1,
        )

        relations = []
        for row, (head_index, tail_index) in enumerate(candidate_pairs):
            relation_index = int(probabilities[row].argmax().item())
            confidence = float(probabilities[row, relation_index].item())
            relation_type = self.id2relation.get(relation_index, "UNKNOWN")
            if relation_type == "NO_RELATION" or confidence < self.relation_threshold:
                continue
            relations.append(
                Relation(
                    entities[head_index],
                    entities[tail_index],
                    relation_type,
                    round(confidence, 4),
                )
            )
        return relations

    @staticmethod
    def _entity_to_dict(entity: Entity) -> Dict[str, Any]:
        return {
            "text": entity.text,
            "type": entity.entity_type,
            "chunk_id": entity.chunk_id,
            "token_span": [entity.token_start, entity.token_end],
            "character_span": [entity.char_start, entity.char_end],
            "page_numbers": entity.page_numbers,
            "sentence_id": entity.sentence_id,
            "confidence": round(entity.confidence, 4),
        }

    @staticmethod
    def _relation_to_dict(relation: Relation) -> Dict[str, Any]:
        evidence_pages = sorted(
            set(relation.head_entity.page_numbers) | set(relation.tail_entity.page_numbers)
        )
        return {
            "head": InferencePipeline._entity_to_dict(relation.head_entity),
            "tail": InferencePipeline._entity_to_dict(relation.tail_entity),
            "relation": relation.relation_type,
            "confidence": round(relation.confidence, 4),
            "evidence_page_numbers": evidence_pages,
        }

    @staticmethod
    def _empty_result(pdf_path: Path) -> Dict[str, Any]:
        return {
            "document": {"filename": pdf_path.name, "total_pages": 0, "total_chunks": 0},
            "finding": {
                "predicted_types": [],
                "predictions": [],
                "predicted_type": None,
                "confidence": 0.0,
                "all_scores": {},
            },
            "entities": [],
            "relationships": [],
        }
