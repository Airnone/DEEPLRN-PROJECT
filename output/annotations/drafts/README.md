# DEEPLRN machine-draft observation labels

These files are review overlays for the 45 structural observation candidates in
`output/observation_candidates/discrepancy_pilot_observations.jsonl`.

They are **not training-ready ground truth**. Every entry requires human
adjudication, and no train/validation/test assignment has been made.

## Files

- `caloocan_city_2020_observation_labels.draft.json` - 11 Caloocan candidates
- `remaining_discrepancy_pilot_observation_labels.draft.json` - 34 candidates
  from Taguig, Pasig, Bacoor, and Naga

## Validated coverage

| Disposition | Count |
| --- | ---: |
| In scope under the current five classes | 3 |
| Outside the current five classes | 39 |
| Human taxonomy decision required | 3 |
| Total | 45 |

Proposed in-scope labels:

- `unliquidated_cash_advance`: 2
- `procurement_irregularity`: 1

The overlays contain 11 exact entity spans and 6 proposed relations. Character
offsets are zero-indexed and half-open within each candidate's
`observation_text`. A validation pass confirmed candidate coverage, offset/text
agreement, entity and relation labels, relation endpoints, and source PDF
checksums.

## Human decisions required before corpus freeze

1. Decide whether unsupported or unverified **account balances** belong inside
   `unsupported_disbursement`. The draft recommendation is no unless the source
   identifies an actual disbursement transaction.
2. Adjudicate Bacoor observation 6 between `procurement_irregularity` and
   `unsupported_disbursement`. The draft primary label is
   `procurement_irregularity` because the observation cites noncompliance with
   GPPB emergency-procurement documentary requirements.
3. Decide whether Naga observations 3 and 4 contain enough detail for an
   existing label or require review of underlying Audit Observation Memoranda.
4. Add a supplemental Naga candidate for the page-12 statement that most
   Notices of Suspension involved unsubmitted Disbursement Vouchers and
   supporting documents. The structural extractor missed this direct candidate.
5. Decide whether the research taxonomy should be expanded to cover the dominant
   corpus families: asset/PPE reconciliation, inventory records, bank/cash
   reconciliation, unsupported balances, and other accounting-control findings.

Do not prepare model records, freeze a split, or start training from these files
until the annotation decisions above are reviewed and the observation-level
record builder is implemented.
