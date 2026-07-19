#!/usr/bin/env python3
"""Evaluate frozen historical models on candidates resolved after the cutoff."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
from collections import Counter, defaultdict

import numpy as np

from validate_ablation_calibration import (
    apply_temperature,
    brier,
    calibrate_temperature,
    ece,
    fit_transform,
    precision_at_k,
    predict,
    train_l2,
    transform,
)


SNAPSHOT_VALUES = [
    "historic_period_days",
    "historic_epoch_btjd",
    "historic_transit_duration_hours",
    "historic_transit_depth_ppm",
    "historic_planet_radius_rearth",
    "historic_stellar_radius_rsun",
    "historic_stellar_teff",
    "historic_stellar_logg",
    "historic_tmag",
    "historic_signal_to_noise",
    "historic_centroid_offset",
]
SNAPSHOT_FEATURES = (
    SNAPSHOT_VALUES
    + [field + "_missing" for field in SNAPSHOT_VALUES]
    + ["historic_sector_count", "historic_pipeline_spoc"]
)
DV_FEATURES = [
    "dv_has_tce",
    "dv_data_impact",
    "dv_data_indur",
    "dv_data_maxmes",
    "dv_data_pradius",
    "dv_data_tdepth",
    "dv_data_tdur",
    "dv_data_tperiod",
    "dv_data_tsnr",
]
XML_FEATURES = [
    "xml_found",
    "xml_oddeventransitdepthcomparisonstatistic_value",
    "xml_oddeventransitdepthcomparisonstatistic_significance",
    "xml_weaksecondary_maxmes",
    "xml_weaksecondary_robuststatistic",
    "xml_bootstrapresults_significance",
    "xml_bootstrapresults_transitcount",
    "xml_differenceimageresults_numberoftransits",
    "xml_planetcandidate_maxmultipleeventsigma",
    "xml_planetcandidate_maxsingleeventsigma",
    "xml_planetcandidate_maxsesinmes",
    "xml_planetcandidate_robuststatistic",
    "xml_planetcandidate_chisquare2",
    "xml_planetcandidate_chisquaregof",
    "xml_alltransitsfit_modelfitsnr",
    "xml_oddtransitsfit_modelfitsnr",
    "xml_eventransitsfit_modelfitsnr",
    "xml_trapezoidalfit_modelfitsnr",
    "xml_depthppm_value",
    "xml_radius_value",
]
FEATURE_SETS = {
    "snapshot_catalog": SNAPSHOT_FEATURES,
    "snapshot_availability": SNAPSHOT_FEATURES + ["dv_has_tce", "xml_found"],
    "snapshot_dv_header": SNAPSHOT_FEATURES + DV_FEATURES,
    "snapshot_dvr_xml": SNAPSHOT_FEATURES + XML_FEATURES,
    "all_cutoff_evidence": SNAPSHOT_FEATURES + DV_FEATURES + XML_FEATURES,
}
KS = (10, 25, 50, 100, 250)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("tic_id", ""), row.get("toi", "")


def merge(
    cohort: list[dict[str, str]],
    dv: list[dict[str, str]],
    xml: list[dict[str, str]],
) -> list[dict[str, str]]:
    dv_by_key = {key(row): row for row in dv}
    xml_by_key = {key(row): row for row in xml}
    output = []
    for source in cohort:
        if source.get("temporal_role") not in {
            "development_resolved_at_cutoff",
            "future_resolved_test",
        }:
            continue
        row = dict(source)
        for field, value in dv_by_key.get(key(source), {}).items():
            if field not in {"tic_id", "toi", "temporal_role"}:
                row[field] = value
        for field, value in xml_by_key.get(key(source), {}).items():
            if field not in {"tic_id", "toi"}:
                row[field] = value
        row.setdefault("dv_has_tce", "0")
        row.setdefault("dv_candidate_match", "0")
        row.setdefault("xml_found", "0")
        output.append(row)
    return output


def labels(rows: list[dict[str, str]], indices: list[int]) -> np.ndarray:
    return np.array([float(rows[index]["temporal_label"]) for index in indices])


def available_features(rows: list[dict[str, str]], requested: list[str]) -> list[str]:
    return [field for field in requested if any(field in row for row in rows)]


def development_split(
    rows: list[dict[str, str]], seed: int, calibration_fraction: float
) -> tuple[list[int], list[int], list[int]]:
    development = [
        index
        for index, row in enumerate(rows)
        if row.get("temporal_role") == "development_resolved_at_cutoff"
    ]
    test = [
        index
        for index, row in enumerate(rows)
        if row.get("temporal_role") == "future_resolved_test"
    ]
    hosts = sorted({rows[index]["tic_id"] for index in development})
    random.Random(seed).shuffle(hosts)
    n_calibration = max(1, round(len(hosts) * calibration_fraction))
    calibration_hosts = set(hosts[:n_calibration])
    train = [index for index in development if rows[index]["tic_id"] not in calibration_hosts]
    calibration = [index for index in development if rows[index]["tic_id"] in calibration_hosts]
    return train, calibration, test


def nll(y: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    return -float(np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    positive = int(y.sum())
    negative = len(y) - positive
    if not positive or not negative:
        return math.nan
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    positive = int(y.sum())
    if not positive:
        return math.nan
    ordered = y[np.argsort(score, kind="mergesort")[::-1]]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / positive)


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    output = {
        "brier": brier(y, probability),
        "ece": ece(y, probability),
        "nll": nll(y, probability),
        "roc_auc": roc_auc(y, probability),
        "average_precision": average_precision(y, probability),
    }
    for k_value in KS:
        result = precision_at_k(y, probability, k_value)
        output[f"precision_at_{k_value}"] = float(result) if result is not None else math.nan
    return output


def fit_model(
    rows: list[dict[str, str]],
    train: list[int],
    calibration: list[int],
    test: list[int],
    requested_features: list[str],
) -> tuple[np.ndarray, float, list[str]]:
    features = available_features(rows, requested_features)
    x_train, stats = fit_transform(rows, train, features)
    weights = train_l2(x_train, labels(rows, train))
    raw_calibration = predict(transform(rows, calibration, features, stats), weights)
    temperature = calibrate_temperature(raw_calibration, labels(rows, calibration))
    raw_test = predict(transform(rows, test, features, stats), weights)
    return apply_temperature(raw_test, temperature), temperature, features


def summarize(repeated: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in repeated:
        grouped[(str(row["model"]), str(row["metric"]))].append(float(row["value"]))
    output = []
    for (model, metric), values in sorted(grouped.items()):
        array = np.array(values)
        output.append(
            {
                "model": model,
                "metric": metric,
                "n_splits": len(array),
                "mean": float(array.mean()),
                "sd": float(array.std(ddof=1)),
                "q025": float(np.quantile(array, 0.025)),
                "q975": float(np.quantile(array, 0.975)),
            }
        )
    return output


def cluster_bootstrap(
    rows: list[dict[str, str]],
    test: list[int],
    predictions: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    host_to_local: dict[str, list[int]] = defaultdict(list)
    for local_index, row_index in enumerate(test):
        host_to_local[rows[row_index]["tic_id"]].append(local_index)
    clusters = [np.array(indices, dtype=int) for _, indices in sorted(host_to_local.items())]
    y = labels(rows, test)
    rng = np.random.default_rng(seed)
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    paired: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _ in range(replicates):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        local = np.concatenate([clusters[index] for index in selected])
        values = {model: metrics(y[local], probability[local]) for model, probability in predictions.items()}
        for model, model_values in values.items():
            for metric, value in model_values.items():
                samples[(model, metric)].append(value)
        full = values["all_cutoff_evidence"]
        for baseline_name in ("snapshot_catalog", "snapshot_availability"):
            baseline = values[baseline_name]
            comparison = f"all_cutoff_evidence_vs_{baseline_name}"
            for metric, baseline_value in baseline.items():
                improvement = (
                    baseline_value - full[metric]
                    if metric in {"brier", "ece", "nll"}
                    else full[metric] - baseline_value
                )
                paired[(comparison, metric)].append(improvement)
    intervals = []
    for (model, metric), values in sorted(samples.items()):
        array = np.array(values)
        intervals.append(
            {
                "model": model,
                "metric": metric,
                "replicates": replicates,
                "mean": float(array.mean()),
                "q025": float(np.quantile(array, 0.025)),
                "q975": float(np.quantile(array, 0.975)),
            }
        )
    improvements = []
    for (comparison, metric), values in sorted(paired.items()):
        array = np.array(values)
        improvements.append(
            {
                "comparison": comparison,
                "metric": metric,
                "replicates": replicates,
                "mean_improvement": float(array.mean()),
                "q025": float(np.quantile(array, 0.025)),
                "q975": float(np.quantile(array, 0.975)),
                "positive_fraction": float((array > 0).mean()),
            }
        )
    return intervals, improvements


def lookup(rows: list[dict[str, object]], model: str, metric: str) -> dict[str, object]:
    return next(row for row in rows if row["model"] == model and row["metric"] == metric)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="data/processed/temporal_cohort_20210617.csv")
    parser.add_argument("--dv", default="data/processed/temporal_dv_20210617.csv")
    parser.add_argument("--xml", default="data/processed/temporal_xml_20210617.csv")
    parser.add_argument("--merged", default="data/processed/temporal_merged_20210617.csv")
    parser.add_argument("--outdir", default="reports/temporal")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=11)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    rows = merge(
        read_csv(pathlib.Path(args.cohort)),
        read_csv(pathlib.Path(args.dv)),
        read_csv(pathlib.Path(args.xml)),
    )
    write_rows(pathlib.Path(args.merged), rows)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    development_hosts = {
        row["tic_id"] for row in rows if row["temporal_role"] == "development_resolved_at_cutoff"
    }
    test_hosts = {row["tic_id"] for row in rows if row["temporal_role"] == "future_resolved_test"}
    product_dates = [
        row.get("dv_product_last_modified", "")
        for row in rows
        if row.get("dv_product_last_modified")
    ]
    coverage_by_role_label: dict[str, dict[str, int]] = {}
    for role in ("development_resolved_at_cutoff", "future_resolved_test"):
        for label in ("0", "1"):
            group = [
                row
                for row in rows
                if row.get("temporal_role") == role and row.get("temporal_label") == label
            ]
            coverage_by_role_label[f"{role}:label_{label}"] = {
                "rows": len(group),
                "dv_matches": sum(row.get("dv_candidate_match") in {"1", "1.0"} for row in group),
                "xml_matches": sum(row.get("xml_found") in {"1", "1.0"} for row in group),
            }
    audit = {
        "rows": len(rows),
        "unique_candidate_keys": len({key(row) for row in rows}),
        "development_rows": sum(row["temporal_role"] == "development_resolved_at_cutoff" for row in rows),
        "development_hosts": len(development_hosts),
        "future_test_rows": sum(row["temporal_role"] == "future_resolved_test" for row in rows),
        "future_test_hosts": len(test_hosts),
        "host_overlap": len(development_hosts & test_hosts),
        "dv_matches_by_role": dict(
            Counter(
                row["temporal_role"]
                for row in rows
                if row.get("dv_candidate_match") in {"1", "1.0"}
            )
        ),
        "xml_matches_by_role": dict(
            Counter(row["temporal_role"] for row in rows if row.get("xml_found") in {"1", "1.0"})
        ),
        "coverage_by_role_label": coverage_by_role_label,
        "latest_product_timestamp": max(product_dates) if product_dates else "",
        "products_after_cutoff": sum(date > "2021-06-17T23:59" for date in product_dates),
        "current_numeric_features_used": [],
    }
    if audit["rows"] != audit["unique_candidate_keys"] or audit["host_overlap"] or audit["products_after_cutoff"]:
        raise RuntimeError(f"temporal integrity failure: {audit}")
    (outdir / "integrity_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    repeated: list[dict[str, object]] = []
    fixed_predictions: dict[str, np.ndarray] = {}
    fixed_test: list[int] = []
    fixed_temperatures: dict[str, float] = {}
    used_features: dict[str, list[str]] = {}
    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        train, calibration, test = development_split(rows, seed, args.calibration_fraction)
        if set(rows[index]["tic_id"] for index in train) & set(rows[index]["tic_id"] for index in calibration):
            raise RuntimeError("development train-calibration host overlap")
        if set(rows[index]["tic_id"] for index in train + calibration) & set(
            rows[index]["tic_id"] for index in test
        ):
            raise RuntimeError("development-test host overlap")
        y_test = labels(rows, test)
        for model, requested in FEATURE_SETS.items():
            probability, temperature, features = fit_model(rows, train, calibration, test, requested)
            values = metrics(y_test, probability)
            for metric, value in values.items():
                repeated.append(
                    {
                        "seed": seed,
                        "model": model,
                        "metric": metric,
                        "value": value,
                        "n_train": len(train),
                        "n_calibration": len(calibration),
                        "n_test": len(test),
                        "temperature": temperature,
                        "n_features": len(features),
                    }
                )
            if seed == args.seed_start:
                fixed_predictions[model] = probability
                fixed_test = test
                fixed_temperatures[model] = temperature
                used_features[model] = features

    summary = summarize(repeated)
    intervals, improvements = cluster_bootstrap(
        rows,
        fixed_test,
        fixed_predictions,
        args.bootstrap_replicates,
        seed=20210617,
    )
    write_rows(outdir / "repeated_development_split_metrics.csv", repeated)
    write_rows(outdir / "repeated_development_split_summary.csv", summary)
    write_rows(outdir / "cluster_bootstrap_intervals.csv", intervals)
    write_rows(outdir / "paired_cluster_bootstrap_improvements.csv", improvements)

    y_fixed = labels(rows, fixed_test)
    prediction_rows: list[dict[str, object]] = []
    for local_index, row_index in enumerate(fixed_test):
        result: dict[str, object] = {
            "tic_id": rows[row_index]["tic_id"],
            "toi": rows[row_index]["toi"],
            "current_disposition": rows[row_index]["current_disposition"],
            "temporal_label": rows[row_index]["temporal_label"],
            "dv_candidate_match": rows[row_index].get("dv_candidate_match", "0"),
            "xml_found": rows[row_index].get("xml_found", "0"),
        }
        for model, probability in fixed_predictions.items():
            result[model + "_probability"] = float(probability[local_index])
        prediction_rows.append(result)
    write_rows(outdir / "fixed_test_predictions.csv", prediction_rows)

    report = {
        "cutoff_date": "2021-06-17",
        "evaluation_design": "resolved-at-cutoff development; unresolved-at-cutoff and later-resolved fixed test",
        "n_repeated_development_splits": args.n_seeds,
        "bootstrap_replicates": args.bootstrap_replicates,
        "development_positive_rate": float(
            np.mean(
                [
                    float(row["temporal_label"])
                    for row in rows
                    if row["temporal_role"] == "development_resolved_at_cutoff"
                ]
            )
        ),
        "future_test_positive_rate": float(y_fixed.mean()),
        "fixed_seed": args.seed_start,
        "fixed_temperatures": fixed_temperatures,
        "features": used_features,
        "integrity": audit,
        "repeated_summary": summary,
        "paired_bootstrap_improvements": improvements,
    }
    (outdir / "temporal_validation_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# True Temporal Validation",
        "",
        "Cutoff: 2021-06-17 archived MAST TOI catalog, Sectors 1-36.",
        "",
        f"Development: {audit['development_rows']} rows on {audit['development_hosts']} hosts. ",
        f"Future-resolved test: {audit['future_test_rows']} rows on {audit['future_test_hosts']} hosts.",
        "",
        "## Repeated development-split results",
        "",
        "The fixed future cohort is unchanged; only the historical train/calibration host partition varies.",
        "",
        "| Model | Brier mean [split interval] | AP mean [split interval] | P@100 mean [split interval] |",
        "|---|---:|---:|---:|",
    ]
    for model in FEATURE_SETS:
        brier_row = lookup(summary, model, "brier")
        ap_row = lookup(summary, model, "average_precision")
        p100_row = lookup(summary, model, "precision_at_100")
        lines.append(
            f"| `{model}` | {brier_row['mean']:.4f} [{brier_row['q025']:.4f}, {brier_row['q975']:.4f}] "
            f"| {ap_row['mean']:.4f} [{ap_row['q025']:.4f}, {ap_row['q975']:.4f}] "
            f"| {p100_row['mean']:.3f} [{p100_row['q025']:.3f}, {p100_row['q975']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Paired TIC-cluster bootstrap",
            "",
            "Positive values favor all cutoff-eligible evidence over the snapshot catalog baseline.",
            "",
            "| Metric | Mean improvement | 95% interval | Positive fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in ("brier", "average_precision", "roc_auc", "precision_at_100"):
        row = next(
            item
            for item in improvements
            if item["metric"] == metric
            and item["comparison"] == "all_cutoff_evidence_vs_snapshot_catalog"
        )
        lines.append(
            f"| `{metric}` | {row['mean_improvement']:.4f} | "
            f"[{row['q025']:.4f}, {row['q975']:.4f}] | {row['positive_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic content beyond product availability",
            "",
            "This comparison controls for the indicator that a cutoff-eligible DV/XML product exists.",
            "",
            "| Metric | Mean improvement | 95% interval | Positive fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in ("brier", "average_precision", "roc_auc", "precision_at_100"):
        row = next(
            item
            for item in improvements
            if item["metric"] == metric
            and item["comparison"] == "all_cutoff_evidence_vs_snapshot_availability"
        )
        lines.append(
            f"| `{metric}` | {row['mean_improvement']:.4f} | "
            f"[{row['q025']:.4f}, {row['q975']:.4f}] | {row['positive_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Integrity and coverage",
            "",
            f"- Development-test TIC overlap: {audit['host_overlap']}.",
            f"- Products dated after the cutoff: {audit['products_after_cutoff']}.",
            f"- Latest accepted product timestamp: {audit['latest_product_timestamp']}.",
            f"- DV matches by role: {audit['dv_matches_by_role']}.",
            f"- XML matches by role: {audit['xml_matches_by_role']}.",
            f"- Development positive rate: {report['development_positive_rate']:.3f}.",
            f"- Future-test positive rate: {report['future_test_positive_rate']:.3f}.",
        ]
    )
    (outdir / "temporal_validation_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
