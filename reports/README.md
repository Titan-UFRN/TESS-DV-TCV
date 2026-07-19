# Tracked report summaries

This directory keeps only concise summaries that correspond to analyses in the
paper. Machine-generated CSV, JSON, PDF, PNG, and SVG artifacts are ignored by
Git and belong in the versioned paper-data release.

- `acceptance/acceptance_analysis_report.md`: strict event association and
  reviewer-requested sensitivity analyses.
- `temporal/temporal_validation_report.md`: original cutoff-based temporal
  benchmark, retained to document the superseded period-only comparison.
- `validation/candidate_integrity_audit.md`: candidate-key and association audit.
- `validation/split_leakage_audit.md`: TIC-host split audit.

The strict acceptance analysis supersedes the period-only temporal result for
scientific interpretation. Preliminary subset experiments, candidate-ranking
policies, and machine-specific execution notes are intentionally not tracked.
