"""
deeplrn.pipeline
~~~~~~~~~~~~~~~~

End-to-end inference pipeline: PDF → preprocessing → model → structured JSON.

Usage
-----
>>> from deeplrn.pipeline import InferencePipeline
>>> pipe = InferencePipeline.from_checkpoint("checkpoints/best_model.pt")
>>> result = pipe.run("path/to/coa_report.pdf")
>>> result["violations"]
[{'type': 'procurement_irregularity', 'confidence': 0.92, 'entities': [...]}]
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from deeplrn.config import (
    CHUNK_CFG,
    EXTRACT_CFG,
    NER_CFG,
    VIOLATION_CFG,
    RELATION_CFG,
    ChunkConfig,
    ExtractionConfig,
)
from deeplrn.preprocessing.pdf_extractor import PDFExtractor, PageData
from deeplrn.preprocessing.chunker import DocumentChunker, TextChunk

logger = logging.getLogger(__name__)


# ── Entity container ─────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A named entity extracted from the document."""

    text: str
    entity_type: str           # e.g. "PERSON", "AMOUNT", "VIOLATION"
    chunk_id: int
    token_start: int
    token_end: int
    page_numbers: List[int] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Relation:
    """A predicted relationship between two entities."""

    head_entity: Entity
    tail_entity: Entity
    relation_type: str         # e.g. "INVOLVES", "AMOUNT_OF", "RESPONSIBLE_FOR"
    confidence: float = 0.0


# ── Pipeline ─────────────────────────────────────────────────────────────────

class InferencePipeline:
    """End-to-end inference: PDF → structured JSON.

    Parameters
    ----------
    model : DeepLRNModel
        A trained multi-task model (on CPU or GPU).
    tokenizer : PreTrainedTokenizerBase
        The tokenizer matching the model's encoder.
    device : str
        'cuda' or 'cpu'.
    extract_cfg : ExtractionConfig
        PDF extraction settings.
    chunk_cfg : ChunkConfig
        Chunking settings.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str = "cpu",
        extract_cfg: ExtractionConfig | None = None,
        chunk_cfg: ChunkConfig | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.extract_cfg = extract_cfg or EXTRACT_CFG
        self.chunk_cfg = chunk_cfg or CHUNK_CFG

        # Tag & label lookups
        self.id2tag = {i: t for i, t in enumerate(NER_CFG.tags)}
        self.id2violation = {i: v for i, v in enumerate(VIOLATION_CFG.labels)}
        self.id2relation = {0: "NO_RELATION"}
        for i, r in enumerate(RELATION_CFG.relation_types, start=1):
            self.id2relation[i] = r

    # ── Class method for loading ─────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> InferencePipeline:
        """Load a pipeline from a saved checkpoint.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to the checkpoint file (saved by Trainer.save_checkpoint).
        device : str
            'cuda', 'cpu', or 'auto' (picks GPU if available).
        """
        from deeplrn.model.deeplrn_model import DeepLRNModel, ModelConfig

        checkpoint_path = Path(checkpoint_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Reconstruct model config
        model_cfg = checkpoint.get("model_config", ModelConfig())
        if isinstance(model_cfg, dict):
            model_cfg = ModelConfig(**model_cfg)

        model = DeepLRNModel(config=model_cfg)
        model.load_state_dict(checkpoint["model_state_dict"])

        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.encoder_name,
            use_fast=True,
        )

        return cls(model=model, tokenizer=tokenizer, device=device)

    # ── Main inference ───────────────────────────────────────────────

    @torch.no_grad()
    def run(self, pdf_path: str | Path) -> Dict[str, Any]:
        """Run full inference on a single PDF.

        Returns
        -------
        dict
            Structured output with document info, entities, violations,
            and relationships.
        """
        pdf_path = Path(pdf_path)
        t0 = time.perf_counter()

        # ── 1. Extract & chunk ───────────────────────────────────────
        extractor = PDFExtractor(pdf_path, config=self.extract_cfg)
        pages = extractor.extract()

        chunker = DocumentChunker(config=self.chunk_cfg, tokenizer=self.tokenizer)
        chunks = chunker.chunk_pages(pages)

        if not chunks:
            logger.warning("No chunks produced from %s", pdf_path.name)
            return self._empty_result(pdf_path)

        # ── 2. Prepare model inputs ──────────────────────────────────
        input_ids, attention_mask = self._prepare_inputs(chunks)

        # ── 3. Forward pass ──────────────────────────────────────────
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # ── 4. Decode predictions ────────────────────────────────────
        entities = self._decode_ner(outputs["ner_logits"], attention_mask, chunks)
        violation = self._decode_violation(outputs["cls_logits"])

        # ── 5. Run relation extraction on predicted entities ─────────
        relations = self._predict_relations(entities, input_ids, attention_mask, chunks)

        # ── 6. Assemble JSON ─────────────────────────────────────────
        t_total = time.perf_counter() - t0

        result = {
            "document": {
                "filename": pdf_path.name,
                "total_pages": len(pages),
                "total_chunks": len(chunks),
                "inference_time_s": round(t_total, 2),
            },
            "violation": {
                "predicted_type": violation["label"],
                "confidence": violation["confidence"],
                "all_scores": violation["all_scores"],
            },
            "entities": [self._entity_to_dict(e) for e in entities],
            "relationships": [self._relation_to_dict(r) for r in relations],
        }

        logger.info(
            "Inference on %s: %d entities, violation=%s (%.2f), %d relations in %.1fs",
            pdf_path.name, len(entities), violation["label"],
            violation["confidence"], len(relations), t_total,
        )
        return result

    def run_and_save(
        self,
        pdf_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Run inference and write the JSON result to disk."""
        pdf_path = Path(pdf_path)
        result = self.run(pdf_path)

        if output_path is None:
            output_path = pdf_path.with_suffix(".json")
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info("Saved result to %s", output_path)
        return result

    # ── Input preparation ────────────────────────────────────────────

    def _prepare_inputs(
        self,
        chunks: List[TextChunk],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize chunks and prepare batched tensors.

        Returns (input_ids, attention_mask) with shape (1, num_chunks, max_tokens).
        """
        max_len = self.chunk_cfg.max_tokens
        num_chunks = len(chunks)

        all_ids = torch.zeros(num_chunks, max_len, dtype=torch.long)
        all_mask = torch.zeros(num_chunks, max_len, dtype=torch.long)

        for i, chunk in enumerate(chunks):
            encoded = self.tokenizer(
                chunk.text,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            all_ids[i] = encoded["input_ids"].squeeze(0)
            all_mask[i] = encoded["attention_mask"].squeeze(0)

        # Add batch dimension: (1, num_chunks, max_tokens)
        return (
            all_ids.unsqueeze(0).to(self.device),
            all_mask.unsqueeze(0).to(self.device),
        )

    # ── NER decoding ─────────────────────────────────────────────────

    def _decode_ner(
        self,
        ner_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        chunks: List[TextChunk],
    ) -> List[Entity]:
        """Decode NER logits into Entity objects.

        Parameters
        ----------
        ner_logits : (1, num_chunks, seq_len, num_tags)
        attention_mask : (1, num_chunks, seq_len)
        chunks : list of TextChunk
        """
        # Remove batch dim
        logits = ner_logits[0]       # (num_chunks, seq_len, num_tags)
        mask = attention_mask[0]     # (num_chunks, seq_len)

        predictions = logits.argmax(dim=-1)  # (num_chunks, seq_len)
        confidences = torch.softmax(logits, dim=-1).max(dim=-1).values

        entities: List[Entity] = []

        for chunk_idx in range(len(chunks)):
            chunk = chunks[chunk_idx]
            pred = predictions[chunk_idx].cpu().tolist()
            conf = confidences[chunk_idx].cpu().tolist()
            chunk_mask = mask[chunk_idx].cpu().tolist()

            # Decode BIO tags into entity spans
            current_entity: dict | None = None

            for tok_idx in range(len(pred)):
                if chunk_mask[tok_idx] == 0:
                    # Padding — close any open entity
                    if current_entity is not None:
                        entities.append(self._build_entity(current_entity, chunk))
                        current_entity = None
                    continue

                tag = self.id2tag.get(pred[tok_idx], "O")

                if tag.startswith("B-"):
                    # Close previous entity if open
                    if current_entity is not None:
                        entities.append(self._build_entity(current_entity, chunk))
                    # Start new entity
                    current_entity = {
                        "type": tag[2:],
                        "chunk_id": chunk_idx,
                        "start": tok_idx,
                        "end": tok_idx + 1,
                        "confidences": [conf[tok_idx]],
                    }
                elif tag.startswith("I-") and current_entity is not None:
                    # Continue entity if types match
                    if tag[2:] == current_entity["type"]:
                        current_entity["end"] = tok_idx + 1
                        current_entity["confidences"].append(conf[tok_idx])
                    else:
                        entities.append(self._build_entity(current_entity, chunk))
                        current_entity = None
                else:
                    # O tag — close any open entity
                    if current_entity is not None:
                        entities.append(self._build_entity(current_entity, chunk))
                        current_entity = None

            # Close entity at chunk boundary
            if current_entity is not None:
                entities.append(self._build_entity(current_entity, chunk))

        return entities

    def _build_entity(self, ent_dict: dict, chunk: TextChunk) -> Entity:
        """Convert a raw entity dict into an Entity dataclass."""
        # Decode the token span back to text
        token_ids = chunk.token_ids[ent_dict["start"]:ent_dict["end"]]
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

        return Entity(
            text=text,
            entity_type=ent_dict["type"],
            chunk_id=ent_dict["chunk_id"],
            token_start=ent_dict["start"],
            token_end=ent_dict["end"],
            page_numbers=chunk.page_numbers,
            confidence=sum(ent_dict["confidences"]) / len(ent_dict["confidences"]),
        )

    # ── Violation decoding ───────────────────────────────────────────

    def _decode_violation(
        self,
        cls_logits: torch.Tensor,
    ) -> Dict[str, Any]:
        """Decode violation classification logits."""
        probs = torch.softmax(cls_logits[0], dim=-1)  # (num_classes,)
        pred_idx = probs.argmax().item()
        pred_label = self.id2violation.get(pred_idx, f"unknown_{pred_idx}")
        pred_conf = probs[pred_idx].item()

        all_scores = {
            self.id2violation.get(i, f"class_{i}"): round(p.item(), 4)
            for i, p in enumerate(probs)
        }

        return {
            "label": pred_label,
            "confidence": round(pred_conf, 4),
            "all_scores": all_scores,
        }

    # ── Relation prediction ──────────────────────────────────────────

    def _predict_relations(
        self,
        entities: List[Entity],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunks: List[TextChunk],
    ) -> List[Relation]:
        """Score all valid entity pairs for relations.

        Entity pairs are only considered if they appear within the same
        chunk or in chunks that are within ``sentence_window`` of each
        other.
        """
        if len(entities) < 2:
            return []

        # Build candidate pairs (within proximity constraint)
        pairs: List[Tuple[int, int]] = []
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i == j:
                    continue
                # Check proximity: chunks within a window
                ci = entities[i].chunk_id
                cj = entities[j].chunk_id
                if abs(ci - cj) <= 2:  # chunks close enough
                    pairs.append((i, j))

        if not pairs:
            return []

        # Get token embeddings for entity spans
        # Re-encode is expensive; instead, use the cached forward pass
        # We run a lightweight forward through just the encoder for span extraction
        flat_ids = input_ids.view(-1, input_ids.shape[-1])
        flat_mask = attention_mask.view(-1, attention_mask.shape[-1])

        token_embs, _ = self.model.encoder(flat_ids, flat_mask)
        # token_embs: (num_chunks, seq_len, hidden)

        head_vecs = []
        tail_vecs = []
        pair_indices = []

        for (hi, ti) in pairs:
            h_ent = entities[hi]
            t_ent = entities[ti]

            h_emb = token_embs[h_ent.chunk_id, h_ent.token_start:h_ent.token_end].mean(dim=0)
            t_emb = token_embs[t_ent.chunk_id, t_ent.token_start:t_ent.token_end].mean(dim=0)

            head_vecs.append(h_emb)
            tail_vecs.append(t_emb)
            pair_indices.append((hi, ti))

        head_tensor = torch.stack(head_vecs)
        tail_tensor = torch.stack(tail_vecs)

        rel_logits = self.model.relation_extractor(head_tensor, tail_tensor)
        rel_probs = torch.softmax(rel_logits, dim=-1)  # (num_pairs, num_relations)

        relations: List[Relation] = []
        for k, (hi, ti) in enumerate(pair_indices):
            pred_idx = rel_probs[k].argmax().item()
            pred_conf = rel_probs[k, pred_idx].item()
            rel_type = self.id2relation.get(pred_idx, "UNKNOWN")

            if rel_type != "NO_RELATION" and pred_conf > 0.3:
                relations.append(Relation(
                    head_entity=entities[hi],
                    tail_entity=entities[ti],
                    relation_type=rel_type,
                    confidence=round(pred_conf, 4),
                ))

        return relations

    # ── Serialisation helpers ────────────────────────────────────────

    @staticmethod
    def _entity_to_dict(entity: Entity) -> Dict[str, Any]:
        return {
            "text": entity.text,
            "type": entity.entity_type,
            "chunk_id": entity.chunk_id,
            "token_span": [entity.token_start, entity.token_end],
            "page_numbers": entity.page_numbers,
            "confidence": round(entity.confidence, 4),
        }

    @staticmethod
    def _relation_to_dict(relation: Relation) -> Dict[str, Any]:
        return {
            "head": {
                "text": relation.head_entity.text,
                "type": relation.head_entity.entity_type,
            },
            "tail": {
                "text": relation.tail_entity.text,
                "type": relation.tail_entity.entity_type,
            },
            "relation": relation.relation_type,
            "confidence": round(relation.confidence, 4),
        }

    @staticmethod
    def _empty_result(pdf_path: Path) -> Dict[str, Any]:
        return {
            "document": {"filename": pdf_path.name, "total_pages": 0, "total_chunks": 0},
            "violation": {"predicted_type": None, "confidence": 0.0, "all_scores": {}},
            "entities": [],
            "relationships": [],
        }
