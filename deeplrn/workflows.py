"""Command-line workflows for preparation, training, evaluation, and inference."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

from deeplrn.checkpoints import load_checkpoint
from deeplrn.config import ChunkConfig, ExtractionConfig
from deeplrn.experiments import PRESETS, get_preset
from deeplrn.model.deeplrn_model import DeepLRNModel, ModelConfig
from deeplrn.pipeline import InferencePipeline
from deeplrn.preprocessing.pdf_extractor import PDFExtractor
from deeplrn.schema import load_annotation
from deeplrn.training.builder import TrainingRecordBuilder, save_training_record
from deeplrn.training.dataset import DeepLRNDataset
from deeplrn.training.trainer import Trainer, TrainingConfig

logger = logging.getLogger("deeplrn")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def prepare_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert annotated COA PDFs to model-ready JSON records."
    )
    parser.add_argument("--annotations", required=True, help="Annotation JSON file or directory")
    parser.add_argument("--pdf-root", default=".", help="Base directory for relative source_pdf paths")
    parser.add_argument("--output", required=True, help="Output record directory")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="deeplrn")
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    annotation_path = Path(args.annotations)
    paths = (
        [annotation_path]
        if annotation_path.is_file()
        else sorted(annotation_path.glob("*.json"))
    )
    if not paths:
        parser.error("no annotation JSON files were found")

    preset = get_preset(args.preset)
    tokenizer_name = args.tokenizer or preset.encoder_name
    max_tokens = args.max_tokens or preset.max_tokens
    overlap = args.overlap if args.overlap is not None else preset.overlap_tokens
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    chunk_config = ChunkConfig(
        max_tokens=max_tokens,
        overlap_tokens=overlap,
        tokenizer_name=tokenizer_name,
        single_sentence_chunks=preset.single_sentence_chunks,
    )
    builder = TrainingRecordBuilder(tokenizer, chunk_config=chunk_config)
    extraction_config = ExtractionConfig(ocr_fallback=not args.no_ocr)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        annotation = load_annotation(path)
        pdf_path = Path(annotation.source_pdf)
        if not pdf_path.is_absolute():
            pdf_path = Path(args.pdf_root) / pdf_path
        pages = PDFExtractor(pdf_path, extraction_config).extract()
        record = builder.build(pages, annotation)
        target = output_dir / f"{annotation.doc_id}.json"
        save_training_record(record, target)
        logger.info("Prepared %s -> %s", pdf_path, target)


def _model_from_args(args: argparse.Namespace) -> DeepLRNModel:
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location="cpu")
        return DeepLRNModel(ModelConfig(**checkpoint["model_config"]))
    preset = get_preset(args.preset)
    options = dict(preset.model_options)
    options["encoder_name"] = args.encoder or preset.encoder_name
    if args.text_only:
        options["use_layout"] = False
    if args.no_document_context:
        options["use_document_context"] = False
    if args.no_page:
        options["use_page_embeddings"] = False
    if args.no_section:
        options["use_section_embeddings"] = False
    if args.no_bbox:
        options["use_bbox_embeddings"] = False
    return DeepLRNModel(ModelConfig(**options))


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train or evaluate DEEPLRN from a split manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="deeplrn")
    parser.add_argument("--encoder")
    parser.add_argument("--resume", help="Versioned checkpoint to resume or evaluate")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--ner-loss-weight", type=float, default=1.0)
    parser.add_argument("--finding-loss-weight", type=float, default=0.5)
    parser.add_argument("--relation-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--no-document-context", action="store_true")
    parser.add_argument("--no-page", action="store_true")
    parser.add_argument("--no-section", action="store_true")
    parser.add_argument("--no-bbox", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    if args.eval_only and not args.resume:
        parser.error("--eval-only requires --resume")

    preset = get_preset(args.preset)
    max_tokens = args.max_tokens or preset.max_tokens
    train_dataset = DeepLRNDataset.from_manifest(
        args.manifest, "train", max_tokens=max_tokens
    )
    eval_dataset = DeepLRNDataset.from_manifest(
        args.manifest, "validation", max_tokens=max_tokens
    )
    if len(train_dataset) == 0:
        parser.error("the manifest contains no training documents")
    if len(eval_dataset) == 0:
        parser.error("the manifest contains no validation documents")

    model = _model_from_args(args)
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        ner_loss_weight=args.ner_loss_weight,
        cls_loss_weight=args.finding_loss_weight,
        rel_loss_weight=args.relation_loss_weight,
        seed=args.seed,
        output_dir=args.output,
    )
    trainer = Trainer(model, train_dataset, eval_dataset, config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    metrics = trainer.evaluate() if args.eval_only else trainer.train()
    print(json.dumps(metrics, indent=2, sort_keys=True))


def infer_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run evidence-linked DEEPLRN PDF inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--relation-threshold", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    pipeline = InferencePipeline.from_checkpoint(
        args.checkpoint,
        device=args.device,
        relation_threshold=args.relation_threshold,
    )
    pipeline.run_and_save(args.input, args.output)
    logger.info("Inference result written to %s", args.output)
