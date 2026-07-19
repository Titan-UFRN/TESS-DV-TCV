# Cutoff-Safe TESS Data-Validation Fusion

Code and lightweight reports for *Cutoff-Safe Tabular Fusion of TESS
Data-Validation Products for Temporal Candidate Vetting*.

This repository evaluates whether tabular TESS Data Validation (DV) headers and
Data Validation Report (DVR) XML summaries available at a historical cutoff
improve the ranking of TESS Objects of Interest by a later catalog disposition.
It is a temporal candidate-vetting benchmark, not a planet-confirmation system
or a validated telescope-allocation policy. Raw flux, pixels, stellar-activity
measurements, and radial-velocity observations are outside its scope.

## Main result and limitations

The frozen cohort uses the MAST TOI catalog released on 2021-06-17. It contains
396 development candidates resolved by that date and 828 candidates that were
unresolved at the cutoff and resolved in the catalog created on 2026-03-14.
Strict event association reduces 639 period-only matches to 545 DV and 510 DVR
matches and excludes diagnostics from 129 ambiguous associations.

Across 20 historical development partitions, strict DV/DVR fusion changes mean
average precision from 0.586 to 0.656 and precision at 100 from 0.600 to 0.740.
The improvement does not extend to the top of the queue: precision at 10 falls
from 0.720 to 0.515. A 300-replicate two-stage host bootstrap gives improvements
of 0.061 [-0.010, 0.147] for average precision and 0.137 [-0.030, 0.335] for
precision at 100; both intervals include zero. Mean temporal ECE is 0.259, so
scores must not be interpreted as transported planet probabilities.

The endpoint is a later TOI disposition. It can reuse evidence related to the
same DV products and is neither independent planet truth nor realized
radial-velocity yield. The repository therefore supports ranking experiments
and failure analysis, not absolute scientific confidence claims.

## Requirements

- Python 3.10 or newer
- NumPy 1.24 or newer
- Optional: `rsvg-convert` for PDF/PNG figure conversion

Create an isolated environment and install the dependency:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Exact inputs

Generated data are intentionally excluded from Git. Restore the matching
processed-input bundle at these paths before reproducing the paper:

```text
data/processed/candidates.csv
data/processed/full_dv_xml_merged.csv
reports/candidate_triage_ranking.csv
data/processed/temporal_cohort_20210617.csv
data/processed/temporal_dv_20210617.csv
data/processed/temporal_xml_20210617.csv
```

The expected sizes, SHA-256 hashes, upstream provenance, and archive-release
instructions are in [`data/README.md`](data/README.md) and
[`data/SHA256SUMS`](data/SHA256SUMS). Verify a restored bundle with:

```sh
python3 scripts/verify_inputs.py
```

The historical catalog and MAST products can also be reconstructed from their
public sources with `make temporal-fetch`. Reconstruction is useful for
provenance checks but is not byte-identical to the paper bundle when the live
outcome catalog has changed. The frozen bundle is therefore the authoritative
input for reproducing reported numbers.

## Reproduce the paper analyses

Run the primary temporal validation and reviewer-requested acceptance analyses:

```sh
make temporal
make acceptance
```

Or run every reporting layer, including the retrospective ablation and
robustness analyses:

```sh
make paper-results
```

Equivalent direct commands for the primary analysis are:

```sh
python3 scripts/validate_temporal.py \
  --cohort data/processed/temporal_cohort_20210617.csv \
  --dv data/processed/temporal_dv_20210617.csv \
  --xml data/processed/temporal_xml_20210617.csv \
  --outdir reports/temporal \
  --n-seeds 20 \
  --bootstrap-replicates 2000

python3 scripts/validate_acceptance_analyses.py \
  --cohort data/processed/temporal_cohort_20210617.csv \
  --dv data/processed/temporal_dv_20210617.csv \
  --xml data/processed/temporal_xml_20210617.csv \
  --outdir reports/acceptance \
  --n-seeds 20 \
  --bootstrap-replicates 300
```

The acceptance analysis enforces period and phase-wrapped epoch agreement,
selected-TCE/DVR planet-number agreement, explicit XML TOI agreement when
present, cutoff-eligible sectors and timestamps, and non-reuse of a strict DVR
product across TOIs. It also evaluates:

- training-only median imputation and field-level missingness indicators;
- complete-case and product-availability strata;
- individual negative-disposition categories;
- a calibrated nonlinear boosted-stump comparator;
- coefficient-regularization sensitivity;
- score-margin abstention with error, calibration, and coverage;
- precision, recall, enrichment, and coverage at fixed queue sizes; and
- a two-stage host bootstrap that refits preprocessing, coefficients, and
  temperature while independently resampling future hosts.

Generated CSV, JSON, PDF, PNG, and SVG artifacts are ignored by Git. The runner
writes `reports/reproducibility_manifest.json` with commands, sizes, and hashes.
The concise tracked summaries in `reports/` identify the publication results.

## Validate the repository

Checks that do not require the processed data bundle:

```sh
make check
```

This compiles all scripts, runs the unit tests, and exercises the complete
command graph in dry-run mode. Network retrieval scripts are kept separate from
deterministic analysis so archive availability cannot silently change a local
validation run.

## Repository layout

```text
scripts/            retrieval, audit, modeling, and reporting programs
tests/              deterministic unit and CLI tests
data/               checksums, provenance notes, and ignored data locations
reports/             concise tracked summaries; generated artifacts are ignored
.github/workflows/  continuous-integration checks
```

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The code is
available under the BSD 3-Clause License; see [`LICENSE`](LICENSE). Public source
products remain subject to the terms of their originating archives.
