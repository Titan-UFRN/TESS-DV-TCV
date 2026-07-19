#!/usr/bin/env python3
"""Run the reproducible experiment/reporting layer from processed inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
from datetime import datetime, timezone


ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], dry_run: bool = False) -> dict[str, object]:
    printable = " ".join(cmd)
    print(f"RUN {printable}")
    if dry_run:
        return {"cmd": cmd, "returncode": 0, "dry_run": True}
    proc = subprocess.run(cmd, cwd=ROOT)
    return {"cmd": cmd, "returncode": proc.returncode, "dry_run": False}


def convert_figures(include_robustness: bool = False, dry_run: bool = False) -> list[dict[str, object]]:
    commands: list[list[str]] = []
    converter = shutil.which("rsvg-convert")
    if not converter:
        print("SKIP rsvg-convert not found; SVG figures were still generated.")
        return []
    pairs = [
        ("reports/validation/reliability_curve.svg", "reports/validation/reliability_curve.pdf", "pdf"),
        ("reports/validation/reliability_curve.svg", "reports/validation/reliability_curve.png", "png"),
        ("reports/validation/subgroup_calibration.svg", "reports/validation/subgroup_calibration.pdf", "pdf"),
        ("reports/validation/subgroup_calibration.svg", "reports/validation/subgroup_calibration.png", "png"),
    ]
    if include_robustness:
        pairs.extend([
            ("reports/robustness/repeated_split_intervals.svg", "reports/robustness/repeated_split_intervals.pdf", "pdf"),
            ("reports/robustness/repeated_split_intervals.svg", "reports/robustness/repeated_split_intervals.png", "png"),
        ])
    for src, dst, fmt in pairs:
        commands.append([converter, "-f", fmt, "-o", dst, src])
    return [run(cmd, dry_run) for cmd in commands]


def write_manifest(path: pathlib.Path, commands: list[dict[str, object]], inputs: list[str], outputs: list[str]) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(ROOT),
        "python": sys.version,
        "commands": commands,
        "inputs": {
            item: {
                "exists": (ROOT / item).exists(),
                "bytes": (ROOT / item).stat().st_size if (ROOT / item).exists() and (ROOT / item).is_file() else None,
                "sha256": sha256(ROOT / item),
            }
            for item in inputs
        },
        "outputs": {
            item: {
                "exists": (ROOT / item).exists(),
                "bytes": (ROOT / item).stat().st_size if (ROOT / item).exists() and (ROOT / item).is_file() else None,
                "sha256": sha256(ROOT / item),
            }
            for item in outputs
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--merged", default="data/processed/full_dv_xml_merged.csv")
    parser.add_argument("--candidate-ranking", default="reports/candidate_triage_ranking.csv")
    parser.add_argument("--manifest", default="reports/reproducibility_manifest.json")
    parser.add_argument("--with-latex", action="store_true", help="Also rebuild paper/main.tex when a paper directory is present.")
    parser.add_argument("--with-robustness", action="store_true", help="Run repeated grouped splits, cluster bootstrap, and ranking stability.")
    parser.add_argument("--with-temporal", action="store_true", help="Run the frozen-cutoff temporal validation experiment.")
    parser.add_argument("--with-acceptance", action="store_true", help="Run strict event matching and reviewer-requested temporal sensitivities.")
    parser.add_argument("--robustness-seeds", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--temporal-cohort", default="data/processed/temporal_cohort_20210617.csv")
    parser.add_argument("--temporal-dv", default="data/processed/temporal_dv_20210617.csv")
    parser.add_argument("--temporal-xml", default="data/processed/temporal_xml_20210617.csv")
    parser.add_argument("--temporal-merged", default="data/processed/temporal_merged_20210617.csv")
    parser.add_argument("--temporal-outdir", default="reports/temporal")
    parser.add_argument("--acceptance-outdir", default="reports/acceptance")
    parser.add_argument("--acceptance-bootstrap-replicates", type=int, default=300)
    parser.add_argument("--temporal-seeds", type=int, default=20)
    parser.add_argument("--temporal-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--skip-latex", action="store_true", help="Deprecated compatibility flag; LaTeX is skipped unless --with-latex is set.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands: list[list[str]] = [
        [
            "python3",
            "scripts/validate_ablation_calibration.py",
            "--merged",
            args.merged,
            "--outdir",
            "reports/validation",
        ],
        [
            "python3",
            "scripts/audit_splits.py",
            "--merged",
            args.merged,
            "--out-json",
            "reports/validation/split_leakage_audit.json",
            "--out-md",
            "reports/validation/split_leakage_audit.md",
        ],
        [
            "python3",
            "scripts/audit_candidate_integrity.py",
            "--candidates",
            args.candidates,
            "--merged",
            args.merged,
            "--out-json",
            "reports/validation/candidate_integrity_audit.json",
            "--out-md",
            "reports/validation/candidate_integrity_audit.md",
        ],
        [
            "python3",
            "scripts/plot_validation.py",
            "--reliability",
            "reports/validation/reliability_curve.csv",
            "--subgroup",
            "reports/validation/subgroup_calibration.csv",
            "--outdir",
            "reports/validation",
        ],
        [
            "python3",
            "scripts/audit_top_candidates.py",
            "--ranking",
            args.candidate_ranking,
            "--top",
            "50",
            "--score-column",
            "triage_score",
            "--out-csv",
            "reports/top50_candidate_audit.csv",
            "--out-json",
            "reports/top50_candidate_audit.json",
            "--out-md",
            "reports/top50_candidate_audit.md",
        ],
        [
            "python3",
            "scripts/revise_candidate_scoring.py",
            "--ranking",
            args.candidate_ranking,
            "--before-summary",
            "reports/top50_candidate_audit.json",
            "--out-ranking",
            "reports/revised_candidate_triage_ranking.csv",
            "--out-summary",
            "reports/revised_triage_comparison.json",
            "--out-md",
            "reports/revised_triage_comparison.md",
            "--top",
            "50",
        ],
        [
            "python3",
            "scripts/audit_top_candidates.py",
            "--ranking",
            "reports/revised_candidate_triage_ranking.csv",
            "--top",
            "50",
            "--score-column",
            "revised_triage_score",
            "--out-csv",
            "reports/revised_top50_candidate_audit.csv",
            "--out-json",
            "reports/revised_top50_candidate_audit.json",
            "--out-md",
            "reports/revised_top50_candidate_audit.md",
        ],
        [
            "python3",
            "scripts/tune_vetting_penalty.py",
            "--merged",
            args.merged,
            "--candidate-ranking",
            args.candidate_ranking,
            "--out-weights",
            "reports/tuned_vetting_penalty.json",
            "--out-ranking",
            "reports/tuned_candidate_triage_ranking.csv",
            "--out-md",
            "reports/tuned_triage_comparison.md",
            "--top",
            "50",
        ],
        [
            "python3",
            "scripts/audit_top_candidates.py",
            "--ranking",
            "reports/tuned_candidate_triage_ranking.csv",
            "--top",
            "50",
            "--score-column",
            "tuned_triage_score",
            "--out-csv",
            "reports/tuned_top50_candidate_audit.csv",
            "--out-json",
            "reports/tuned_top50_candidate_audit.json",
            "--out-md",
            "reports/tuned_top50_candidate_audit.md",
        ],
        [
            "python3",
            "scripts/write_policy_case_studies.py",
            "--original",
            args.candidate_ranking,
            "--hand",
            "reports/revised_candidate_triage_ranking.csv",
            "--tuned",
            "reports/tuned_candidate_triage_ranking.csv",
            "--out",
            "reports/policy_case_studies.md",
            "--top",
            "10",
        ],
    ]

    if args.with_robustness:
        commands.append([
            "python3",
            "scripts/validate_statistical_robustness.py",
            "--merged",
            args.merged,
            "--outdir",
            "reports/robustness",
            "--n-seeds",
            str(args.robustness_seeds),
            "--bootstrap-replicates",
            str(args.bootstrap_replicates),
        ])
        commands.append([
            "python3",
            "scripts/plot_robustness.py",
            "--summary",
            "reports/robustness/repeated_split_summary.csv",
            "--out",
            "reports/robustness/repeated_split_intervals.svg",
        ])

    if args.with_temporal:
        commands.append([
            "python3",
            "scripts/validate_temporal.py",
            "--cohort",
            args.temporal_cohort,
            "--dv",
            args.temporal_dv,
            "--xml",
            args.temporal_xml,
            "--merged",
            args.temporal_merged,
            "--outdir",
            args.temporal_outdir,
            "--n-seeds",
            str(args.temporal_seeds),
            "--bootstrap-replicates",
            str(args.temporal_bootstrap_replicates),
        ])

    if args.with_acceptance:
        commands.append([
            "python3",
            "scripts/validate_acceptance_analyses.py",
            "--cohort", args.temporal_cohort,
            "--dv", args.temporal_dv,
            "--xml", args.temporal_xml,
            "--outdir", args.acceptance_outdir,
            "--n-seeds", str(args.temporal_seeds),
            "--bootstrap-replicates", str(args.acceptance_bootstrap_replicates),
        ])

    inputs = [
        args.candidates,
        args.merged,
        args.candidate_ranking,
    ]
    if args.with_temporal or args.with_acceptance:
        inputs.extend([
            args.temporal_cohort,
            args.temporal_dv,
            args.temporal_xml,
        ])
    outputs = [
        "reports/validation/ablation_metrics.csv",
        "reports/validation/reliability_curve.csv",
        "reports/validation/subgroup_calibration.csv",
        "reports/validation/split_leakage_audit.json",
        "reports/validation/candidate_integrity_audit.json",
        "reports/top50_candidate_audit.json",
        "reports/revised_triage_comparison.json",
        "reports/revised_top50_candidate_audit.json",
        "reports/tuned_vetting_penalty.json",
        "reports/tuned_candidate_triage_ranking.csv",
        "reports/tuned_top50_candidate_audit.json",
        "reports/policy_case_studies.md",
    ]
    if args.with_robustness:
        outputs.extend([
            "reports/robustness/repeated_split_summary.csv",
            "reports/robustness/paired_cluster_bootstrap_improvements.csv",
            "reports/robustness/paired_policy_improvements.csv",
            "reports/robustness/policy_repeated_summary.csv",
            "reports/robustness/candidate_ranking_stability.csv",
            "reports/robustness/statistical_robustness_summary.json",
            "reports/robustness/statistical_robustness_report.md",
            "reports/robustness/repeated_split_intervals.svg",
        ])
    if args.with_temporal:
        temporal_outdir = pathlib.Path(args.temporal_outdir)
        outputs.extend([
            str(temporal_outdir / "integrity_audit.json"),
            str(temporal_outdir / "repeated_development_split_metrics.csv"),
            str(temporal_outdir / "repeated_development_split_summary.csv"),
            str(temporal_outdir / "cluster_bootstrap_intervals.csv"),
            str(temporal_outdir / "paired_cluster_bootstrap_improvements.csv"),
            str(temporal_outdir / "fixed_test_predictions.csv"),
            str(temporal_outdir / "temporal_validation_summary.json"),
            str(temporal_outdir / "temporal_validation_report.md"),
            args.temporal_merged,
        ])
    if args.with_acceptance:
        acceptance_outdir = pathlib.Path(args.acceptance_outdir)
        outputs.extend([
            str(acceptance_outdir / "event_level_provenance.csv"),
            str(acceptance_outdir / "repeated_sensitivity_summary.csv"),
            str(acceptance_outdir / "missingness_stratified_metrics.csv"),
            str(acceptance_outdir / "label_category_sensitivity.csv"),
            str(acceptance_outdir / "coefficient_regularization_sensitivity.csv"),
            str(acceptance_outdir / "abstention_margin_tradeoff.csv"),
            str(acceptance_outdir / "two_stage_host_bootstrap_summary.csv"),
            str(acceptance_outdir / "acceptance_analysis_summary.json"),
            str(acceptance_outdir / "acceptance_analysis_report.md"),
        ])
    if shutil.which("rsvg-convert"):
        outputs.extend([
            "reports/validation/reliability_curve.pdf",
            "reports/validation/reliability_curve.png",
            "reports/validation/subgroup_calibration.pdf",
            "reports/validation/subgroup_calibration.png",
        ])
        if args.with_robustness:
            outputs.extend([
                "reports/robustness/repeated_split_intervals.pdf",
                "reports/robustness/repeated_split_intervals.png",
            ])

    records: list[dict[str, object]] = []
    for cmd in commands:
        record = run(cmd, args.dry_run)
        records.append(record)
        if record["returncode"] != 0:
            write_manifest(ROOT / args.manifest, records, inputs, outputs)
            return int(record["returncode"])
    records.extend(convert_figures(args.with_robustness, args.dry_run))
    if args.with_latex and not args.skip_latex:
        inputs.append("paper/main.tex")
        outputs.append("paper/main.pdf")
        records.append(run(["latexmk", "-cd", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "paper/main.tex"], args.dry_run))
    write_manifest(ROOT / args.manifest, records, inputs, outputs)
    return 0 if all(record["returncode"] == 0 for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
