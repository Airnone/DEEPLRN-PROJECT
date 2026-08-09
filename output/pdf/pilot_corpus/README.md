# DEEPLRN COA pilot corpus v1

Collected on 2026-08-07 from public Commission on Audit reports published by
official Philippine local-government websites. The corpus is deliberately small:
it is intended to validate annotation, leakage-resistant splitting, preprocessing,
OCR, and training mechanics before a larger research collection is attempted.

## Scope

- 8 Annual Audit Report PDFs
- 5 independent LGU groups
- 711 pages
- Calendar years 2020-2024
- Regions I, III, IV-B, V, and IX
- Entity mix: one province, two cities, and two municipalities
- Seven text-bearing PDFs and one image-only OCR case

The files are under `pdfs/`. `corpus_manifest.json` records official source
pages, download URLs, checksums, page counts, extraction quality, and known
limitations. Original official ZIP packages used for La Union and Taytay are
preserved under `source_archives/`.

## Important training constraint

This is a real document corpus, but it is not labeled training data yet. A COA
Annual Audit Report normally contains several audit observations that can belong
to different finding categories. The current `AnnotatedDocument` schema assigns
exactly one `finding_label` to the entire source PDF. Applying one class to a
whole AAR would create noisy and potentially misleading supervision.

Before annotation, use one of these designs:

1. Recommended: segment each AAR into audit-observation units and assign one
   finding label, evidence spans, entities, and relations to each unit while
   retaining the parent LGU and report year for grouped splitting.
2. Alternative: change the model and schema to multi-label document
   classification, while still annotating evidence spans per finding.

The first design aligns most closely with the existing single-label classifier.
No train/validation/test manifest should be frozen until that representation is
implemented and reviewed.

## Quality notes

- `plaridel_2024_aar.pdf` is a 102-page image-only scan and requires OCR.
  The Python `pytesseract` package is installed, but the Tesseract executable
  is not currently available on this machine and must be installed before
  preprocessing that report.
- The Molave files are concise official AAR/executive-report PDFs of 5-6 pages.
- The Naga file is a concise 12-page official AAR PDF.
- Taytay 2020 and 2021 were distributed as ordered PDF parts. The corpus copies
  merge those PDF parts in filename order. Each source archive also contains a
  Part III `.doc` file that is preserved in the ZIP but is not present in the
  merged PDF.
- La Union 2024 was extracted from the official ZIP. The accompanying financial
  statements remain in the preserved source archive as an `.xlsx` file.
- La Union 2022 and 2023 were inspected but excluded because the official
  archives primarily contain editable Office files rather than a complete AAR
  PDF suitable for the current pipeline.

## Next step

Implement observation-level corpus records, then annotate a small calibration
set from all five LGU groups. Use dual review for category and span decisions
before scaling annotation. The five-group pilot is sufficient for pipeline
testing, but not for defensible held-out performance claims; the corpus should be
expanded to more LGUs before final model evaluation.
