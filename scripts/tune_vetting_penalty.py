#!/usr/bin/env python3
"""Tune vetting penalties on strict-label calibration data and rank candidates."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import pathlib
from collections import Counter

import numpy as np

from train_full_triage import (
    BASE_FEATURES,
    apply_temperature,
    calibrate_temperature,
    fit_transform,
    predict,
    read_csv,
    split_by_host,
    to_float,
    train_l2_logistic,
    transform,
    y_for,
)
from revise_candidate_scoring import PENALTY_BY_NOTE, needs_manual_review, risk_notes


GROUPS = {
    "large_radius": {"large_radius", "moderately_large_radius"},
    "deep": {"deep", "very_deep"},
    "scene": {"crowded", "many_contaminants"},
    "odd_even": {"odd_even_signal"},
    "secondary": {"secondary_signal"},
    "period": {"ultra_short_period"},
    "missing_context": {"no_xml", "no_dv_tce"},
}

GRID = [0.20, 0.35, 0.50, 0.65, 0.80, 1.00]


def write_rows(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rv_utility(row: dict[str, object]) -> float:
    direct = to_float(row.get("rv_utility"))
    if math.isfinite(direct):
        return max(0.0, direct)
    base = to_float(row.get("score_filtered_followup"))
    return max(0.0, base) if math.isfinite(base) else 0.0


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return float("nan")
    order = np.argsort(score)[::-1][: min(k, len(y))]
    return float(y[order].mean()) if len(order) else float("nan")


def risk_mask(rows: list[dict[str, object]], idx: list[int]) -> np.ndarray:
    return np.array([1.0 if needs_manual_review(risk_notes({k: str(v) for k, v in rows[i].items()})) else 0.0 for i in idx])


def penalty_from_groups(notes: list[str], weights: dict[str, float]) -> float:
    penalty = 1.0
    note_set = set(notes)
    for group, group_notes in GROUPS.items():
        if note_set & group_notes:
            penalty *= weights[group]
    return max(0.02, min(1.0, penalty))


def hand_penalty(notes: list[str]) -> float:
    penalty = 1.0
    for note in notes:
        penalty *= PENALTY_BY_NOTE.get(note, 1.0)
    return max(0.02, min(1.0, penalty))


def penalties(rows: list[dict[str, object]], idx: list[int], weights: dict[str, float] | None) -> np.ndarray:
    vals = []
    for i in idx:
        row = {k: str(v) for k, v in rows[i].items()}
        notes = risk_notes(row)
        vals.append(hand_penalty(notes) if weights is None else penalty_from_groups(notes, weights))
    return np.array(vals, dtype=np.float64)


def metric_summary(y: np.ndarray, scores: np.ndarray, risks: np.ndarray, ks: tuple[int, ...] = (25, 50, 100, 250)) -> dict[str, object]:
    out: dict[str, object] = {}
    for k in ks:
        top = np.argsort(scores)[::-1][: min(k, len(scores))]
        out[f"precision_at_{k}"] = float(y[top].mean()) if len(top) else None
        out[f"manual_review_rate_at_{k}"] = float(risks[top].mean()) if len(top) else None
        out[f"manual_review_count_at_{k}"] = int(risks[top].sum()) if len(top) else 0
    return out


def objective(y: np.ndarray, scores: np.ndarray, risks: np.ndarray) -> float:
    order50 = np.argsort(scores)[::-1][: min(50, len(scores))]
    order100 = np.argsort(scores)[::-1][: min(100, len(scores))]
    if len(order100) == 0:
        return -1e9
    p50 = float(y[order50].mean()) if len(order50) else 0.0
    p100 = float(y[order100].mean())
    risk100 = float(risks[order100].mean())
    return p100 + 0.25 * p50 - 0.20 * risk100


def tune_weights(y: np.ndarray, base_scores: np.ndarray, risks: np.ndarray, note_matrix: list[list[str]]) -> tuple[dict[str, float], float]:
    best_weights: dict[str, float] = {group: 1.0 for group in GROUPS}
    best_obj = -1e9
    group_names = list(GROUPS)
    for combo in itertools.product(GRID, repeat=len(group_names)):
        weights = dict(zip(group_names, combo))
        penalty_vals = np.array([penalty_from_groups(notes, weights) for notes in note_matrix], dtype=np.float64)
        score = base_scores * penalty_vals
        obj = objective(y, score, risks)
        if obj > best_obj:
            best_obj = obj
            best_weights = weights
    return best_weights, best_obj


def note_matrix(rows: list[dict[str, object]], idx: list[int]) -> list[list[str]]:
    return [risk_notes({k: str(v) for k, v in rows[i].items()}) for i in idx]


def candidate_rows_with_policy(rows: list[dict[str, str]], weights: dict[str, float]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        notes = risk_notes(row)
        penalty = penalty_from_groups(notes, weights)
        p_planet = to_float(row.get("calibrated_planet_probability"))
        utility = rv_utility(row)
        if not math.isfinite(p_planet):
            p_planet = 0.0
        tuned_score = p_planet * utility * penalty
        item: dict[str, object] = dict(row)
        item["tuned_vetting_penalty"] = penalty
        item["tuned_vetting_risk_notes"] = ";".join(notes) if notes else "none"
        item["tuned_vetting_risk_note_count"] = len(notes)
        item["tuned_needs_manual_review"] = int(needs_manual_review(notes))
        item["tuned_triage_score"] = tuned_score
        out.append(item)
    out.sort(key=lambda row: to_float(row.get("tuned_triage_score")), reverse=True)
    return out


def candidate_summary(rows: list[dict[str, object]], top: int) -> dict[str, object]:
    selected = rows[:top]
    risk_counts = Counter()
    for row in selected:
        for note in str(row.get("tuned_vetting_risk_notes", "")).split(";"):
            if note and note != "none":
                risk_counts[note] += 1
    return {
        "n": len(selected),
        "class_counts": dict(Counter(str(row.get("strict_planet_class", "")) for row in selected)),
        "needs_manual_review": sum(int(row.get("tuned_needs_manual_review", 0)) for row in selected),
        "risk_note_counts": dict(risk_counts),
        "median_tuned_triage_score": median([to_float(row.get("tuned_triage_score")) for row in selected]),
        "median_calibrated_planet_probability": median([to_float(row.get("calibrated_planet_probability")) for row in selected]),
        "median_expected_k_ms": median([to_float(row.get("expected_k_ms")) for row in selected]),
        "median_tuned_vetting_penalty": median([to_float(row.get("tuned_vetting_penalty")) for row in selected]),
    }


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def write_markdown(path: pathlib.Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tuned = report["strict_test_metrics"]["tuned"]  # type: ignore[index]
    original = report["strict_test_metrics"]["original"]  # type: ignore[index]
    hand = report["strict_test_metrics"]["hand_penalty"]  # type: ignore[index]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Tuned Vetting Penalty\n\n")
        f.write("The penalty groups were tuned on the strict-label calibration split and evaluated on the strict-label test split.\n\n")
        f.write("## Tuned Weights\n\n")
        for group, value in report["tuned_weights"].items():  # type: ignore[union-attr]
            f.write(f"- `{group}`: {value}\n")
        f.write("\n## Strict Test Metrics\n\n")
        f.write("| Policy | P@50 | P@100 | P@250 | Manual@50 | Manual@100 | Manual@250 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for name, metrics in (("Original", original), ("Hand penalty", hand), ("Tuned penalty", tuned)):
            f.write(
                f"| {name} | {metrics['precision_at_50']:.3f} | {metrics['precision_at_100']:.3f} | "
                f"{metrics['precision_at_250']:.3f} | {metrics['manual_review_count_at_50']} | "
                f"{metrics['manual_review_count_at_100']} | {metrics['manual_review_count_at_250']} |\n"
            )
        f.write("\n## Candidate Top-50 Summary\n\n")
        cand = report["candidate_top50"]  # type: ignore[index]
        f.write(f"- Manual-review flags: {cand['needs_manual_review']}\n")
        f.write(f"- Median calibrated p(planet): {cand['median_calibrated_planet_probability']:.3f}\n")
        f.write(f"- Median expected K: {cand['median_expected_k_ms']:.2f} m/s\n")
        f.write(f"- Median tuned score: {cand['median_tuned_triage_score']:.3f}\n")
        f.write("\nRisk notes:\n\n")
        for note, count in sorted(cand["risk_note_counts"].items()):  # type: ignore[index]
            f.write(f"- `{note}`: {count}\n")
        f.write("\n## Interpretation\n\n")
        f.write("The tuned policy is selected on strict-label validation data, so it is easier to defend than the hand-set policy. ")
        f.write("It should be reported alongside the original and hand-penalized policies because the ranking target is a policy choice: a high-amplitude vetting queue and a cleaner RV-priority queue answer different follow-up questions.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", default="data/processed/full_dv_xml_merged.csv")
    parser.add_argument("--candidate-ranking", default="reports/candidate_triage_ranking.csv")
    parser.add_argument("--out-weights", default="reports/tuned_vetting_penalty.json")
    parser.add_argument("--out-ranking", default="reports/tuned_candidate_triage_ranking.csv")
    parser.add_argument("--out-md", default="reports/tuned_triage_comparison.md")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    rows = read_csv(pathlib.Path(args.merged))
    feature_rows: list[dict[str, object]] = [dict(row) for row in rows]
    features = [feat for feat in BASE_FEATURES if any(feat in row for row in feature_rows)]
    train_idx, cal_idx, test_idx = split_by_host(feature_rows)
    train_rows = [feature_rows[i] for i in train_idx]
    x_train, stats = fit_transform(train_rows, features)
    y_train = y_for(feature_rows, train_idx)
    weights_model = train_l2_logistic(x_train, y_train, l2=0.01)
    p_cal_raw = predict(transform([feature_rows[i] for i in cal_idx], features, stats), weights_model)
    y_cal = y_for(feature_rows, cal_idx)
    temp = calibrate_temperature(p_cal_raw, y_cal)
    p_cal = apply_temperature(p_cal_raw, temp)
    p_test = apply_temperature(predict(transform([feature_rows[i] for i in test_idx], features, stats), weights_model), temp)
    y_test = y_for(feature_rows, test_idx)

    cal_base = p_cal * np.array([rv_utility(feature_rows[i]) for i in cal_idx], dtype=np.float64)
    test_base = p_test * np.array([rv_utility(feature_rows[i]) for i in test_idx], dtype=np.float64)
    cal_risk = risk_mask(feature_rows, cal_idx)
    test_risk = risk_mask(feature_rows, test_idx)
    best_weights, best_objective = tune_weights(y_cal, cal_base, cal_risk, note_matrix(feature_rows, cal_idx))

    test_tuned_penalty = penalties(feature_rows, test_idx, best_weights)
    test_hand_penalty = penalties(feature_rows, test_idx, None)
    strict_metrics = {
        "original": metric_summary(y_test, test_base, test_risk),
        "hand_penalty": metric_summary(y_test, test_base * test_hand_penalty, test_risk),
        "tuned": metric_summary(y_test, test_base * test_tuned_penalty, test_risk),
    }

    candidates = candidate_rows_with_policy(read_csv(pathlib.Path(args.candidate_ranking)), best_weights)
    write_rows(pathlib.Path(args.out_ranking), candidates)
    report: dict[str, object] = {
        "tuning_split": "calibration",
        "evaluation_split": "test",
        "seed": 11,
        "temperature": temp,
        "features": features,
        "tuned_weights": best_weights,
        "calibration_objective": best_objective,
        "strict_test_metrics": strict_metrics,
        "candidate_top50": candidate_summary(candidates, args.top),
    }
    pathlib.Path(args.out_weights).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(pathlib.Path(args.out_md), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
