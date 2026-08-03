"""
Smoke test: instantiate the full DeepLRN model and run a single forward
pass with synthetic data to verify all modules connect correctly.

Usage: python tests/test_integration.py
"""

from __future__ import annotations

import sys
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("test_integration")


def main():
    logger.info("=" * 60)
    logger.info("DeepLRN Integration Smoke Test")
    logger.info("=" * 60)

    # ── 1. Import all modules ────────────────────────────────────────
    logger.info("Importing modules...")
    from deeplrn.model.deeplrn_model import DeepLRNModel, ModelConfig
    from deeplrn.model.encoder import ChunkEncoder
    from deeplrn.model.doc_transformer import DocumentTransformer
    from deeplrn.model.heads.ner import NERHead
    from deeplrn.model.heads.classifier import ViolationClassifier
    from deeplrn.model.heads.relation import RelationExtractor
    logger.info("  ✓ All model modules imported successfully")

    from deeplrn.preprocessing import PDFExtractor, DocumentChunker, TextChunk
    logger.info("  ✓ Preprocessing modules imported successfully")

    from deeplrn.pipeline import InferencePipeline
    logger.info("  ✓ Pipeline module imported successfully")

    from deeplrn.training.dataset import create_dummy_dataset, DeepLRNDataset, collate_documents
    from deeplrn.training.trainer import Trainer, TrainingConfig
    logger.info("  ✓ Training modules imported successfully")

    # ── 2. Instantiate model ─────────────────────────────────────────
    logger.info("\nInstantiating DeepLRNModel...")
    config = ModelConfig(
        encoder_name="roberta-base",
        freeze_layers=6,  # freeze half the layers for faster test
    )
    model = DeepLRNModel(config=config)
    logger.info("  ✓ Model instantiated")

    # ── 3. Count parameters ──────────────────────────────────────────
    param_counts = model.count_parameters()
    for name, counts in param_counts.items():
        if isinstance(counts, dict):
            logger.info(
                "  %s: %s total, %s trainable",
                name,
                f"{counts['total']:,}",
                f"{counts['trainable']:,}",
            )

    # ── 4. Create synthetic batch ────────────────────────────────────
    logger.info("\nCreating synthetic batch...")
    batch_size = 2
    num_chunks = 4
    seq_len = 384

    input_ids = torch.randint(0, 50265, (batch_size, num_chunks, seq_len))
    attention_mask = torch.ones(batch_size, num_chunks, seq_len, dtype=torch.long)
    chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool)
    # Mark last chunk of second doc as padding
    chunk_mask[1, -1] = False

    ner_labels = torch.randint(0, 17, (batch_size, num_chunks, seq_len))
    ner_labels[:, :, 0] = -100    # ignore special tokens
    violation_labels = torch.randint(0, 5, (batch_size,))

    # Relation triples: (head_chunk, head_start, head_end, tail_chunk, tail_start, tail_end, rel_label)
    relation_triples = [
        [(0, 5, 8, 0, 15, 18, 1), (1, 10, 13, 1, 20, 23, 2)],  # doc 0
        [(0, 3, 6, 0, 30, 33, 0)],                                # doc 1
    ]

    logger.info("  ✓ Synthetic batch created: batch=%d, chunks=%d, seq_len=%d",
                batch_size, num_chunks, seq_len)

    # ── 5. Forward pass ──────────────────────────────────────────────
    logger.info("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_mask=chunk_mask,
            ner_labels=ner_labels,
            violation_labels=violation_labels,
            relation_triples=relation_triples,
        )

    # ── 6. Verify outputs ────────────────────────────────────────────
    logger.info("\nVerifying output shapes...")

    ner_logits = outputs["ner_logits"]
    assert ner_logits.shape == (batch_size, num_chunks, seq_len, 17), \
        f"NER logits shape mismatch: {ner_logits.shape}"
    logger.info("  ✓ NER logits:       %s", tuple(ner_logits.shape))

    cls_logits = outputs["cls_logits"]
    assert cls_logits.shape == (batch_size, 5), \
        f"Classification logits shape mismatch: {cls_logits.shape}"
    logger.info("  ✓ Class logits:     %s", tuple(cls_logits.shape))

    rel_logits = outputs["rel_logits"]
    assert rel_logits is not None
    assert rel_logits.shape[1] == 4, \
        f"Relation logits dim mismatch: {rel_logits.shape}"
    logger.info("  ✓ Relation logits:  %s  (3 pairs total)", tuple(rel_logits.shape))

    assert "loss" in outputs
    loss = outputs["loss"]
    logger.info("  ✓ Total loss:       %.4f", loss.item())
    logger.info("    NER loss:         %.4f", outputs["ner_loss"].item())
    logger.info("    CLS loss:         %.4f", outputs["cls_loss"].item())
    logger.info("    REL loss:         %.4f", outputs["rel_loss"].item())

    # ── 7. Test gradient flow ────────────────────────────────────────
    logger.info("\nTesting gradient flow...")
    model.train()
    outputs2 = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        chunk_mask=chunk_mask,
        ner_labels=ner_labels,
        violation_labels=violation_labels,
        relation_triples=relation_triples,
    )
    outputs2["loss"].backward()

    # Check that at least some gradients are non-zero
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients flowing!"
    logger.info("  ✓ Gradients are flowing correctly")

    # ── Done! ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("ALL INTEGRATION TESTS PASSED ✓")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
