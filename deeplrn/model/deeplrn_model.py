"""
deeplrn.model.deeplrn_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unified multi-task model that wires together:

1. **ChunkEncoder** — RoBERTa-base encodes each chunk into token embeddings
2. **DocumentTransformer** — 2-layer cross-chunk attention enriches CLS representations
3. **NERHead** — token-level BIO tagging
4. **ViolationClassifier** — document-level violation category prediction
5. **RelationExtractor** — bilinear scoring of entity-pair relations

All three heads are trained jointly via a weighted multi-task loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from deeplrn.config import FINDING_CFG, NER_CFG, RELATION_CFG

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Full model hyperparameters."""

    # Encoder
    encoder_name: str = "roberta-base"
    freeze_layers: int = 0
    encoder_uses_2d_positions: bool = False

    # Document Transformer
    doc_num_layers: int = 2
    doc_num_heads: int = 8
    doc_dropout: float = 0.1
    max_chunks: int = 128
    use_document_context: bool = True

    # Token-level layout fusion and ablations
    use_layout: bool = True
    use_page_embeddings: bool = True
    use_section_embeddings: bool = True
    use_bbox_embeddings: bool = True
    max_pages: int = 512
    max_sections: int = 256
    bbox_bins: int = 1024

    # Head dimensions (derived from encoder, but can override)
    hidden_size: int = 768

    # Task sizes
    num_ner_tags: int = 17       # len(NER_CFG.tags)
    num_violation_classes: int = len(FINDING_CFG.labels)  # legacy field name
    num_relations: int = 4       # 3 types + NO_RELATION

    # Head dropout
    head_dropout: float = 0.1

    # Loss weights
    ner_loss_weight: float = 1.0
    cls_loss_weight: float = 0.5
    rel_loss_weight: float = 0.5


class DeepLRNModel(nn.Module):
    """Unified multi-task model for COA audit report analysis.

    Forward pass:
    1. Encode each chunk with RoBERTa → token embeddings + CLS
    2. Pass chunk CLS embeddings through Document Transformer → enriched CLS
    3. Run NER head on token embeddings (enriched with document context)
    4. Run Violation Classifier on pooled document embedding
    5. Run Relation Extractor on entity-pair embeddings
    6. Compute weighted multi-task loss

    Parameters
    ----------
    config : ModelConfig
        Model hyperparameters.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        if cfg.num_violation_classes != len(FINDING_CFG.labels):
            raise ValueError(
                "num_violation_classes must match the canonical finding schema: "
                f"expected {len(FINDING_CFG.labels)}, got {cfg.num_violation_classes}"
            )

        # ── Lazy imports so this module doesn't fail if subagent files
        #    haven't been written yet during development ──
        from deeplrn.model.encoder import ChunkEncoder
        from deeplrn.model.doc_transformer import DocumentTransformer
        from deeplrn.model.heads.ner import NERHead
        from deeplrn.model.heads.classifier import ViolationClassifier
        from deeplrn.model.heads.relation import RelationExtractor
        from deeplrn.model.layout import LayoutFeatureEncoder

        # ── Sub-modules ──────────────────────────────────────────────
        self.encoder = ChunkEncoder(
            model_name=cfg.encoder_name,
            freeze_layers=cfg.freeze_layers,
        )
        # Ensure hidden_size matches what the encoder actually provides
        cfg.hidden_size = self.encoder.hidden_size

        self.layout_encoder = LayoutFeatureEncoder(
            hidden_size=cfg.hidden_size,
            max_pages=cfg.max_pages,
            max_sections=cfg.max_sections,
            bbox_bins=cfg.bbox_bins,
            use_page=cfg.use_layout and cfg.use_page_embeddings,
            use_section=cfg.use_layout and cfg.use_section_embeddings,
            use_bbox=cfg.use_layout and cfg.use_bbox_embeddings,
            dropout=cfg.head_dropout,
        )

        self.doc_transformer = DocumentTransformer(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.doc_num_layers,
            num_heads=cfg.doc_num_heads,
            dropout=cfg.doc_dropout,
            max_chunks=cfg.max_chunks,
        )

        self.ner_head = NERHead(
            hidden_size=cfg.hidden_size,
            num_tags=cfg.num_ner_tags,
            dropout=cfg.head_dropout,
        )

        self.classifier = ViolationClassifier(
            hidden_size=cfg.hidden_size,
            num_classes=cfg.num_violation_classes,
            dropout=cfg.head_dropout,
        )

        self.relation_extractor = RelationExtractor(
            hidden_size=cfg.hidden_size,
            num_relations=cfg.num_relations,
            dropout=cfg.head_dropout,
        )

        logger.info(
            "DeepLRNModel initialised: encoder=%s, hidden=%d, "
            "doc_layers=%d, ner_tags=%d, viol_classes=%d, rel_types=%d",
            cfg.encoder_name, cfg.hidden_size,
            cfg.doc_num_layers, cfg.num_ner_tags,
            cfg.num_violation_classes, cfg.num_relations,
        )

    # ── Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
        ner_labels: torch.Tensor | None = None,
        finding_labels: torch.Tensor | None = None,
        violation_labels: torch.Tensor | None = None,
        relation_triples: list | None = None,
        page_ids: torch.Tensor | None = None,
        section_ids: torch.Tensor | None = None,
        bboxes: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        """Full forward pass with optional loss computation.

        Parameters
        ----------
        input_ids : Tensor, shape (batch, num_chunks, seq_len)
            Token IDs for each chunk in each document.
        attention_mask : Tensor, shape (batch, num_chunks, seq_len)
            Attention masks for RoBERTa.
        chunk_mask : Tensor, shape (batch, num_chunks), optional
            Boolean mask — True for real chunks, False for padding.
        ner_labels : Tensor, shape (batch, num_chunks, seq_len), optional
            NER tag indices (-100 for padding / ignored tokens).
        finding_labels : Tensor, shape (batch, num_classes), optional
            Multi-hot audit-finding targets.
        relation_triples : list of lists, optional
            For each document in the batch, a list of
            ``(head_chunk, head_start, head_end, tail_chunk, tail_start, tail_end, rel_label)``
            tuples.

        Returns
        -------
        dict with keys:
            - ``ner_logits``: (batch, num_chunks, seq_len, num_tags)
            - ``cls_logits``: (batch, num_classes)
            - ``rel_logits``: (total_pairs, num_relations) or None
            - ``loss``: scalar (only if labels provided)
            - ``ner_loss``, ``cls_loss``, ``rel_loss``: per-task losses
        """
        batch_size, num_chunks, seq_len = input_ids.shape
        device = input_ids.device
        cfg = self.config

        # ── 1. Encode all chunks ─────────────────────────────────────
        # Flatten to (batch * num_chunks, seq_len) for RoBERTa
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)

        if cfg.encoder_uses_2d_positions:
            encoder_boxes = (
                bboxes.view(-1, seq_len, 4)
                if bboxes is not None
                else torch.zeros(
                    (*flat_ids.shape, 4), dtype=torch.long, device=flat_ids.device
                )
            )
            token_embeddings, cls_embeddings = self.encoder(
                flat_ids, flat_mask, bboxes=encoder_boxes
            )
        else:
            token_embeddings, cls_embeddings = self.encoder(flat_ids, flat_mask)
        # token_embeddings: (batch*num_chunks, seq_len, hidden)
        # cls_embeddings:   (batch*num_chunks, hidden)

        hidden = cfg.hidden_size

        if cfg.use_layout:
            token_embeddings = self.layout_encoder(
                token_embeddings,
                page_ids=page_ids.view(-1, seq_len) if page_ids is not None else None,
                section_ids=(
                    section_ids.view(-1, seq_len) if section_ids is not None else None
                ),
                bboxes=bboxes.view(-1, seq_len, 4) if bboxes is not None else None,
            )

        # Reshape back to document structure
        token_embs = token_embeddings.view(batch_size, num_chunks, seq_len, hidden)
        cls_embs = cls_embeddings.view(batch_size, num_chunks, hidden)

        # ── 2. Document Transformer ──────────────────────────────────
        enriched_cls = (
            self.doc_transformer(cls_embs, chunk_mask)
            if cfg.use_document_context
            else cls_embs
        )
        # enriched_cls: (batch, num_chunks, hidden)

        # ── 3. NER Head ─────────────────────────────────────────────
        # Flatten chunks for per-token prediction
        flat_token_embs = token_embs.view(-1, seq_len, hidden)
        # Pass enriched CLS as document context: (batch*num_chunks, hidden)
        flat_doc_ctx = enriched_cls.reshape(-1, hidden)

        ner_logits = self.ner_head(flat_token_embs, flat_doc_ctx)
        # ner_logits: (batch*num_chunks, seq_len, num_tags)
        ner_logits = ner_logits.view(batch_size, num_chunks, seq_len, cfg.num_ner_tags)

        # ── 4. Violation Classifier ──────────────────────────────────
        # Pool enriched CLS across chunks (masked mean)
        if chunk_mask is not None:
            mask_expanded = chunk_mask.unsqueeze(-1).float()  # (batch, num_chunks, 1)
            doc_embedding = (enriched_cls * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            doc_embedding = enriched_cls.mean(dim=1)
        # doc_embedding: (batch, hidden)

        cls_logits = self.classifier(doc_embedding)
        # cls_logits: (batch, num_classes)

        # ── 5. Relation Extraction ───────────────────────────────────
        rel_logits = None
        if relation_triples is not None:
            rel_logits = self._extract_relations(
                token_embs, enriched_cls, relation_triples,
            )

        # ── 6. Compute losses ────────────────────────────────────────
        result: Dict[str, Any] = {
            "ner_logits": ner_logits,
            "cls_logits": cls_logits,
            "rel_logits": rel_logits,
        }

        total_loss = torch.tensor(0.0, device=device)
        has_loss = False

        if ner_labels is not None and bool((ner_labels != -100).any()):
            flat_ner_logits = ner_logits.view(-1, cfg.num_ner_tags)
            flat_ner_labels = ner_labels.view(-1)
            ner_loss = self.ner_head.compute_loss(
                flat_ner_logits.view(-1, seq_len, cfg.num_ner_tags),
                flat_ner_labels.view(-1, seq_len),
                attention_mask.view(-1, seq_len),
            )
            result["ner_loss"] = ner_loss
            total_loss = total_loss + cfg.ner_loss_weight * ner_loss
            has_loss = True

        if finding_labels is not None and violation_labels is not None:
            raise ValueError("pass finding_labels or violation_labels, not both")
        effective_finding_labels = (
            finding_labels if finding_labels is not None else violation_labels
        )
        if effective_finding_labels is not None:
            if effective_finding_labels.ndim == 1:
                effective_finding_labels = torch.nn.functional.one_hot(
                    effective_finding_labels.long(),
                    num_classes=cfg.num_violation_classes,
                ).float()
            cls_loss = self.classifier.compute_loss(cls_logits, effective_finding_labels)
            result["cls_loss"] = cls_loss
            total_loss = total_loss + cfg.cls_loss_weight * cls_loss
            has_loss = True

        if relation_triples is not None and rel_logits is not None:
            rel_labels = self._gather_relation_labels(relation_triples, device)
            if rel_labels is not None and len(rel_labels) > 0:
                rel_loss = self.relation_extractor.compute_loss(rel_logits, rel_labels)
                result["rel_loss"] = rel_loss
                total_loss = total_loss + cfg.rel_loss_weight * rel_loss
                has_loss = True

        if has_loss:
            result["loss"] = total_loss

        return result

    # ── Relation extraction helpers ──────────────────────────────────

    def _extract_relations(
        self,
        token_embs: torch.Tensor,
        enriched_cls: torch.Tensor,
        relation_triples: list,
    ) -> torch.Tensor | None:
        """Extract entity-pair representations and score relations.

        Parameters
        ----------
        token_embs : (batch, num_chunks, seq_len, hidden)
        enriched_cls : (batch, num_chunks, hidden)  — unused here but available
        relation_triples : list of lists of tuples
            ``(head_chunk, head_start, head_end, tail_chunk, tail_start, tail_end, rel_label)``

        Returns
        -------
        Tensor (total_pairs, num_relations) or None if no pairs.
        """
        head_vecs = []
        tail_vecs = []

        for doc_idx, triples in enumerate(relation_triples):
            for (hc, hs, he, tc, ts, te, _) in triples:
                # Average token embeddings over the entity span
                h_emb = token_embs[doc_idx, hc, hs:he].mean(dim=0)
                t_emb = token_embs[doc_idx, tc, ts:te].mean(dim=0)
                head_vecs.append(h_emb)
                tail_vecs.append(t_emb)

        if not head_vecs:
            return None

        head_tensor = torch.stack(head_vecs)  # (total_pairs, hidden)
        tail_tensor = torch.stack(tail_vecs)  # (total_pairs, hidden)

        return self.relation_extractor(head_tensor, tail_tensor)

    @staticmethod
    def _gather_relation_labels(
        relation_triples: list,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Collect relation labels from all documents into a flat tensor."""
        labels = []
        for triples in relation_triples:
            for (*_, rel_label) in triples:
                labels.append(rel_label)
        if not labels:
            return None
        return torch.tensor(labels, dtype=torch.long, device=device)

    # ── Utility ──────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters per sub-module."""
        counts = {}
        for name, module in [
            ("encoder", self.encoder),
            ("layout_encoder", self.layout_encoder),
            ("doc_transformer", self.doc_transformer),
            ("ner_head", self.ner_head),
            ("classifier", self.classifier),
            ("relation_extractor", self.relation_extractor),
        ]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable}
        counts["model_total"] = {
            "total": sum(c["total"] for c in counts.values() if isinstance(c, dict)),
            "trainable": sum(c["trainable"] for c in counts.values() if isinstance(c, dict)),
        }
        return counts
