# Observation candidate extraction - Step 1

This directory contains the structural extraction output for the
`deeplrn-coa-discrepancy-pilot-v1` corpus. No finding labels, entity labels,
relations, train/validation/test assignments, or model predictions have been
added.

## Files

- `discrepancy_pilot_observations.jsonl`: one unreviewed candidate per line
- `discrepancy_pilot_observations.summary.json`: extraction counts and QA flags

The run produced 45 candidates from five documents:

| LGU | Candidates |
| --- | ---: |
| Taguig City | 14 |
| Caloocan City | 11 |
| Pasig City | 6 |
| Bacoor City, Cavite | 10 |
| Naga City | 4 |

## Candidate record

Each JSONL record contains:

- stable observation and parent-document IDs;
- source PDF, LGU, and report year;
- source section and original numbered item, when available;
- one-indexed PDF page range and complete page list;
- separately reconstructed observation and recommendation text;
- extraction method, review status, warnings, and source metadata.

Every record has `review_status: "unreviewed"`. The extractor deliberately does
not emit `finding_label`. A candidate is a section of audit-report text, not a
determination of misconduct, intent, liability, or guilt.

## QA notes

- Bacoor's first four observations share one numbered recommendation block.
  The extractor maps the matching recommendation item back to each observation
  and adds `shared_recommendation_block` for reviewer visibility.
- Naga's first summary exception is unnumbered and is marked
  `unnumbered_observation`.
- Two Taguig candidates have `recommendation_not_detected`; both remain in the
  output for review rather than being silently discarded.
- The source text is retained as extracted. OCR/text-layer spelling defects are
  not silently corrected because reviewers need traceability to the PDF.

## Reproduce

```powershell
python -m deeplrn.preprocessing.observation_extractor `
  --manifest output/pdf/discrepancy_pilot_corpus/corpus_manifest.json `
  --output output/observation_candidates/discrepancy_pilot_observations.jsonl
```

OCR fallback is disabled by default for this command. Pass `--ocr` only when
Tesseract is installed and a corpus contains image-only observation pages.

