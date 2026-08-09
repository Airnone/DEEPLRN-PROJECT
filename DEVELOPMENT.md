# DEEPLRN Complete Model Development Record

This document is the living engineering record for implementing the revised
DEEPLRN proposal. Each step records what was made, why it was made that way,
how it was verified, and what remains dependent on real COA data or human
annotation.

## Definition of complete

The repository will be considered **software-complete and training-ready** when
it can:

1. extract text, page evidence, sections, and normalized word coordinates from
   COA PDFs, with OCR fallback;
2. convert a documented human-annotation format into aligned token labels,
   finding labels, relation candidates, and layout features;
3. create leakage-resistant train, validation, and test manifests by holding
   out whole LGUs and keeping near-duplicate documents in one partition;
4. train the multi-task RoBERTa and document-Transformer model with NER,
   finding-classification, and relation losses;
5. optionally fuse page, section, and bounding-box features and disable each
   feature for ablation experiments;
6. calculate real NER, classification, relation, tuple, calibration, and
   evidence-provenance metrics;
7. save and reload self-describing checkpoints;
8. run PDF-to-JSON inference with neutral labels, confidence, and source-page
   evidence; and
9. pass unit and end-to-end tests without requiring a trained production
   checkpoint.

The project will be considered **research-complete** only after the external,
data-dependent work is also finished: acquiring the intended COA corpus,
performing and adjudicating human annotation, training all seeds and baselines,
and reporting held-out-LGU results. Code alone cannot honestly satisfy those
requirements.

## Implementation plan

| Step | Deliverable | Status |
| --- | --- | --- |
| 1 | Completion criteria and living development record | Complete |
| 2 | Stable schemas, neutral labels, configuration, and checkpoints | Complete |
| 3 | Annotation conversion and training-ready dataset preparation | Complete |
| 4 | Whole-LGU splitting and near-duplicate leakage controls | Complete |
| 5 | Layout-aware multi-task model and evidence provenance | Complete |
| 6 | Relation candidates, metrics, trainer, and command-line workflows | Complete |
| 7 | Baselines, ablations, and end-to-end tests | Complete |
| 8 | Final verification and data-dependent handoff | Complete |

## Step 1 - Completion criteria and living record

### What was made

- A concrete distinction between software completeness and research
  completeness.
- A dependency-ordered implementation plan.
- This development record, which will be updated after every verified step.

### Why it was made this way

The original README marked modules as "Done" when their classes existed, even
though there was no annotated corpus, trained checkpoint, real evaluation, or
working checkpoint contract. Completion is therefore defined in terms of
observable end-to-end capabilities and verified outputs, not file presence.

### Result

The project now has an auditable target state. Subsequent steps can be evaluated
against explicit contracts, and data-dependent limitations will remain visible
instead of being hidden behind placeholder metrics.

### Verification

- The plan covers every implementation commitment in the revised proposal.
- No claim of model accuracy or research completion is made before real data is
  available.

## Step 2 - Stable schemas, neutral labels, and checkpoints

### What was made

- `deeplrn/schema.py` defines the canonical entity, finding, and relation
  labels and validates document annotations.
- `VIOLATION` and `RESPONSIBLE_FOR` are migrated to the neutral `FINDING` and
  `ASSOCIATED_WITH` labels. Explicit aliases keep old prototype annotations
  readable during migration.
- The fifth category is now `contractor_related_concern` rather than
  `suspicious_contractor_activity`.
- `deeplrn/checkpoints.py` creates versioned checkpoints containing model
  configuration, training configuration, label order, optimizer state,
  scheduler state, progress, and metrics.
- Training and inference now share the same checkpoint loader and validator.

### Why it was made this way

Label order is part of a trained model: changing it after training silently
changes the meaning of output logits. Saving the label schema and model
configuration inside every checkpoint prevents that failure. Neutral labels
also keep the software aligned with the proposal's rule that extraction is not
a legal judgment.

### Result

Annotations now fail early when spans overlap, labels are unsupported, IDs are
duplicated, or relations reference missing entities. New checkpoints are
self-describing and incompatible label sets are rejected instead of being
loaded silently.

### Verification

- Python compilation succeeded for the package and tests.
- Direct smoke checks confirmed legacy-label migration and checkpoint
  validation.
- Unit tests were added for schema validation and checkpoint contracts; the
  full pytest run is deferred until development dependencies are installed.

## Step 3 - Annotation conversion and training-ready data

### What was made

- Preprocessing now exports full page text, page dimensions, headers, word
  boxes, chunk token IDs, and exact chunk source offsets.
- Chunk text now preserves the exact source substring so annotation and token
  offsets use the same coordinate system.
- `TrainingRecordBuilder` converts character-span annotations into padded token
  IDs, attention masks, BIO labels, page IDs, section IDs, normalized bounding
  boxes, sentence IDs, and document-level token offsets.
- Positive annotated relations and sentence-window-constrained `NO_RELATION`
  candidates are converted into the span tuples consumed by the relation head.
- `DeepLRNDataset` and its collator now carry layout and evidence tensors while
  retaining temporary aliases for older finding-label field names.
- The invalid setuptools backend was corrected, dependencies were installed,
  and tokenizer-dependent unit tests were made offline and deterministic.

### Why it was made this way

Human annotators work most reliably with visible character spans and page
evidence, while neural models train on subword-token indices. A single
deterministic conversion step prevents every training script from inventing a
slightly different alignment rule. Exact source substrings and stored global
token offsets also make predictions traceable to the original report.

Negative relation candidates are created during conversion because training
only on annotated positive pairs would teach the relation head that every pair
has a relation. The 12-sentence limit is applied to the actual sentence IDs,
not approximated with chunk distance.

### Result

Preprocessing output can now be converted into the exact tensors expected by
the training loader. Layout and evidence information survives the complete
data path even before the model begins using those tensors.

### Verification

- A deterministic synthetic annotation was aligned to BIO labels, normalized
  layout coordinates, and the expected `ASSOCIATED_WITH` relation class.
- All current tests pass: **41 passed**.

## Step 4 - Leakage-resistant corpus splitting

### What was made

- `deeplrn/splitting.py` builds deterministic train, validation, and test
  manifests from model-ready records.
- Every year belonging to an LGU is assigned as one group.
- Five-word shingles, a 64-value MinHash signature, locality-sensitive hashing,
  and exact Jaccard confirmation identify likely copied reports efficiently.
- When near duplicates cross LGU boundaries, the affected LGUs are joined into
  one split component so copied text cannot leak across partitions.
- The manifest records its seed, target ratios, actual counts, duplicate
  threshold, confirmed pairs, cluster IDs, and source paths.
- A `deeplrn-split` command-line entry point was added.

### Why it was made this way

An LGU-year split does not stop a municipality's repeated language from
appearing in both training and testing. Whole-LGU grouping fixes the direct
case; duplicate clustering also handles copied templates shared by multiple
LGUs. MinHash limits expensive exact comparisons to plausible candidates,
while final Jaccard verification keeps the decision interpretable.

### Result

The repository can now create an auditable held-out-LGU experiment without
depending on ad hoc folder placement. The same seed and corpus produce the same
manifest.

### Verification

- Tests confirm that multiple years from one LGU share a split.
- Tests confirm that near-duplicate reports from different LGUs share a split.
- Determinism is verified.
- All tests pass: **44 passed**.

## Step 5 - Layout-aware model and evidence provenance

### What was made

- A layout encoder embeds page IDs, section IDs, and normalized word-box
  coordinates, then fuses their average with RoBERTa token representations
  through a residual projection.
- Independent configuration switches disable page, section, or bounding-box
  inputs, while one master switch provides a text-only ablation.
- Training collation and model forwarding now carry the layout tensors without
  changing the document-Transformer, NER, finding, or relation head contracts.
- Inference now uses the same feature builder as training, decodes BIO spans
  back to global character offsets, and preserves source pages and sentences.
- Overlap-created duplicate entities are merged by their type and exact source
  span. Relation candidates use the configured sentence distance and inference
  JSON includes evidence-bearing head and tail entities.

### Why it was made this way

COA reports use page position, tables, and repeated section structures to
communicate meaning that plain text order can lose. Token-level fusion lets
every task use those signals without replacing the pretrained language
encoder. Residual fusion keeps the original token representation available,
which makes optimization safer and makes the text-only ablation exact.

Training and inference must construct identical features; otherwise evaluation
can measure a preprocessing mismatch instead of the model. Character offsets
and page IDs are therefore treated as first-class outputs rather than being
reconstructed from displayed prediction text.

### Result

The main model is now genuinely layout-aware and can be compared fairly with
page-only, section-only, box-only, and text-only variants. Every extracted
entity and relation can point back to its source evidence in the PDF.

### Verification

- Unit tests confirm layout fusion preserves tensor shape and changes enabled
  representations.
- The master ablation returns the original token representation exactly.
- A tiny offline encoder exercises the full multi-task forward pass with all
  layout tensors.
- All tests pass: **47 passed**.

## Step 6 - Metrics, trainer, and runnable workflows

### What was made

- The trainer now computes exact global-character entity span precision,
  recall, and F1; finding accuracy and macro metrics; multiclass Brier score and
  expected calibration error; positive-only relation metrics; evidence-aware
  tuple F1; and conditional evidence-page accuracy.
- `NO_RELATION` is explicitly class 0 and true negatives do not inflate
  relation F1. Wrong positive types count as both a false positive and false
  negative.
- The training loop now seeds data shuffling and Torch, handles the final
  partial gradient-accumulation step, synchronizes task weights with the model,
  evaluates at intervals and epoch boundaries, and writes best, periodic, and
  final checkpoints.
- Resume restores optimizer, scheduler, completed epoch, global step, metrics,
  and the best selection score.
- Four installed commands now cover annotation preparation, leakage-resistant
  splitting, training/evaluation, and PDF inference: `deeplrn-prepare`,
  `deeplrn-split`, `deeplrn-train`, and `deeplrn-infer`.
- Model records retain relation-candidate entity IDs and annotated evidence
  pages so provenance quality can be evaluated rather than merely displayed.

### Why it was made this way

Token accuracy is misleading for sparse entity extraction, and relation
accuracy is dominated by `NO_RELATION`. Exact structured sets make every
reported true positive correspond to a complete, source-addressable item.
Macro finding F1 exposes weak categories, while calibration measures whether a
displayed confidence can be interpreted as reliability.

Durable progress and label order belong inside the checkpoint because a weight
file alone cannot be resumed or interpreted safely. The command workflows use
the same builders and loaders as tests so a research run does not depend on an
undocumented notebook.

### Result

An annotated corpus can now move from PDFs to deterministic records, manifests,
joint training, real validation metrics, resumable checkpoints, and
evidence-linked inference without custom integration code. Placeholder zero
metrics have been removed.

### Verification

- Metric unit tests cover malformed BIO recovery, exact character spans,
  classification, calibration, positive-only relation scoring, and provenance
  tuples.
- Manifest selection is tested independently.
- A deterministic trainable model exercises evaluation, one optimizer epoch,
  best/final saving, and checkpoint resume.
- All workflow parsers load and print their expected options.

## Step 7 - Baselines, ablations, and statistical comparisons

### What was made

- `deeplrn/experiments.py` defines fixed presets for the full DEEPLRN model,
  local RoBERTa, Longformer, and text-plus-box LayoutLMv3.
- The encoder adapter can pass normalized bounding boxes directly to
  LayoutLMv3, while the Longformer preset uses long text windows.
- A trained TF-IDF bigram plus class-balanced linear-SVM finding baseline can
  be saved with its label order, seed, and validation metrics.
- Command flags disable document context, all layout, individual layout
  sources, or any auxiliary task loss for controlled ablations.
- `deeplrn/statistics.py` implements deterministic LGU-cluster bootstrap F1
  intervals, paired cluster permutation tests, and Holm correction.
- The Python requirement was corrected to 3.10 because the package uses modern
  union type syntax throughout.

### Why it was made this way

Encoder families use different vocabularies, so each preset records its own
tokenizer, window, overlap, and model behavior instead of treating token IDs as
portable. Ablations use the same main implementation and change one declared
switch, minimizing confounding code differences.

Reports from one LGU are correlated. Resampling individual findings would
produce overly narrow intervals, so uncertainty and paired comparisons operate
on whole LGU clusters. Holm correction controls the family-wise error created
by comparing several baselines.

### Result

The proposed comparisons can be launched reproducibly once records are
prepared for each encoder. Statistical code is ready to consume actual
per-LGU counts, but deliberately emits no invented experimental result.

### Verification

- Tests distinguish every neural preset and train/persist the SVM on a small
  deterministic corpus.
- Cluster bootstrap reproducibility, paired comparison direction, and Holm
  adjusted values are tested.
- The two-dimensional encoder path is exercised with an offline tiny encoder.

## Step 8 - Final verification and handoff

### What was made

- `README.md` was replaced with the accurate installation, annotation,
  preparation, split, training, resume, evaluation, inference, baseline, and
  ablation workflow.
- The README explicitly separates a complete software implementation from an
  unperformed research experiment and removes the prototype's accusatory output
  examples.
- This record now contains the rationale, implementation method, result, and
  verification evidence for every development step.

### Why it was made this way

A runnable model repository can still be scientifically incomplete. The
handoff therefore states both facts: the pipeline is implemented and tested,
but model quality is unknown until real labels and held-out runs exist. The
project does not require or implement a human-review/usability study; supervised
labels are nevertheless necessary for defensible training and testing.

### Result

The repository meets the software-complete definition at the start of this
document. The remaining work is data acquisition and experimental execution,
not missing model plumbing.

### Verification

- `python -m compileall -q deeplrn tests` succeeds.
- `python -m pytest -q` succeeds: **58 passed**.
- Ruff passes on every newly added evaluation, experiment, statistics,
  workflow, trainer, and associated test file.
- All four neural presets serialize correctly, and prepare/train/infer command
  parsers pass smoke checks.

## Data-dependent work that remains

1. Expand the collected COA pilot beyond the current five-LGU source set under
   the applicable access and redistribution rules.
2. Review the 45 observation candidates extracted from the discrepancy-focused
   pilot and produce reliable entity, finding, relation, and evidence-page
   labels. Candidate extraction is complete, but it is not human annotation.
3. Prepare separate record sets for encoder families whose tokenizers differ.
4. Freeze the split manifest, thresholds, seeds, and hyperparameter search.
5. Train all fixed seeds, run the held-out-LGU test once, and calculate the
   clustered intervals and corrected paired comparisons.
6. Report measured failures and limitations. Until then, accuracy and
   superiority remain unknown.
