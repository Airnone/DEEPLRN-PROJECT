# DEEPLRN discrepancy-focused pilot corpus

Collected on 2026-08-07 for a manageable, observation-rich training pilot. This
revision explicitly includes Taguig City, a Cavite city, two additional Metro
Manila cities, and retains Naga City.

## Included strata

| LGU | Geography | Document | Selection signal |
| --- | --- | --- | --- |
| Taguig City | Metro Manila | COA 2020 executive summary | Qualified opinion; incomplete physical count and unreconciled PPE records |
| Caloocan City | Metro Manila | COA 2020 executive summary | Qualified opinion; unreconciled treasury records and PPE discrepancies |
| Pasig City | Metro Manila | COA 2020 executive summary | Qualified opinion; incomplete inventory and unreconciled PPE records |
| Bacoor City | Cavite, Region IV-A | COA 2020 executive summary | Qualified opinion; difference between PPE ledger and physical-count report |
| Naga City | Camarines Sur, Region V | Full COA 2024 AAR | Qualified opinion; accounting/property-record discrepancies for PPE and inventory |

This is a deliberate urban comparison set. The three same-year Metro Manila
reports reduce year-specific confounding; Bacoor adds a fast-growing Cavite city
outside NCR; Naga preserves the requested regional comparator and a newer full
AAR. All five have embedded text, so the first annotation pass does not depend
on OCR.

## Neutral use of “discrepancy”

Here, *discrepancy* means an exception, difference, deficiency, or uncertainty
documented by the Commission on Audit. It is a corpus-selection signal, not a
finding of fraud, corruption, personal culpability, or criminal conduct. Audit
opinions concern the presentation of financial statements, and COA states that
opinions are not rankings or grades.

## Provenance

The four 2020 executive summaries originated on COA's public report catalog.
The current COA site blocks automated retrieval of those legacy file paths, so
the preserved COA-origin PDFs were retrieved through the Internet Archive. The
manifest records both the original COA URL and the exact archive URL. Naga's
2024 full AAR was downloaded from the official Naga City government website and
matches the copy retained in pilot corpus v1.

Executive summaries are kept as such and are not represented as complete AARs.
They are useful for a small finding-classification pilot because they contain
high-density summaries of the audit opinion, observations, and recommendations.
The Naga document provides a full-report counterexample for segmentation tests.

## Training status

The PDFs are verified source documents, but the corpus is not labeled training
data yet. Do not assign one finding class to an entire report. Segment the
documents into individual audit-observation units, attach evidence spans, and
group train/validation/test splits by LGU so observations from the same city do
not leak across splits.

Candidate labels for the calibration pass include:

- `asset_record_reconciliation`
- `cash_or_bank_reconciliation`
- `inventory_count_or_record`
- `unliquidated_cash_advance`
- `unsupported_or_unverified_balance`
- `other_control_or_compliance_observation`

The candidate labels must be confirmed against the project's annotation schema
and independently reviewed before model training.

