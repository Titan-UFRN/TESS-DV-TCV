#!/usr/bin/env python3
"""Train regularized/calibrated strict-label baseline and rank PC/APC candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
from collections import defaultdict

import numpy as np


BASE_FEATURES = [
    "period_days",
    "planet_radius_rearth",
    "stellar_mass_msun",
    "stellar_radius_rsun",
    "tmag",
    "transit_depth_ppm_or_percent_raw",
    "transit_duration_hours",
    "stellar_teff",
    "stellar_logg",
    "binary_like_risk",
    "dv_has_tce",
    "dv_tce_count",
    "dv_data_impact",
    "dv_data_indur",
    "dv_data_maxmes",
    "dv_data_pradius",
    "dv_data_tdepth",
    "dv_data_tdur",
    "dv_data_tperiod",
    "dv_data_tsnr",
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


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


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


def to_float(value: object) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def merge(
    candidates: list[dict[str, str]],
    dv: list[dict[str, str]],
    xml: list[dict[str, str]],
    period_match_rtol: float = 0.01,
) -> list[dict[str, object]]:
    dv_by_tic = {row["tic_id"]: row for row in dv if row.get("tic_id")}
    xml_by_tic = {row["tic_id"]: row for row in xml if row.get("tic_id")}
    out: list[dict[str, object]] = []
    for cand in candidates:
        row: dict[str, object] = dict(cand)
        tic = cand.get("tic_id", "")
        dv_row = dv_by_tic.get(tic, {})
        xml_row = xml_by_tic.get(tic, {})

        host_has_tce = str(dv_row.get("dv_has_tce", "")) in {"1", "1.0"}
        row["dv_host_has_tce"] = int(host_has_tce)
        row["dv_host_tce_count"] = dv_row.get("dv_tce_count", "")
        row["xml_host_found"] = int(str(xml_row.get("xml_found", "")) in {"1", "1.0"})

        candidate_period = to_float(cand.get("period_days"))
        dv_period = to_float(dv_row.get("dv_data_tperiod"))
        period_rel_error = math.nan
        if math.isfinite(candidate_period) and math.isfinite(dv_period) and max(abs(candidate_period), abs(dv_period)) > 0:
            period_rel_error = abs(candidate_period - dv_period) / max(abs(candidate_period), abs(dv_period))

        if not host_has_tce:
            match_reason = "no_host_tce"
        elif not math.isfinite(candidate_period):
            match_reason = "missing_candidate_period"
        elif not math.isfinite(dv_period):
            match_reason = "missing_dv_period"
        elif period_rel_error > period_match_rtol:
            match_reason = "period_mismatch"
        else:
            match_reason = "matched"

        dv_match = match_reason == "matched"
        row["dv_candidate_match"] = int(dv_match)
        row["dv_candidate_period_rel_error"] = period_rel_error
        row["dv_match_reason"] = match_reason
        row["dv_has_tce"] = 0
        row["dv_tce_count"] = 0

        if dv_match:
            for key, value in dv_row.items():
                if key.startswith("dv_"):
                    row[key] = value
            row["dv_has_tce"] = 1

        same_product = (
            dv_match
            and xml_row
            and str(xml_row.get("dv_selected_sector", "")) == str(dv_row.get("dv_selected_sector", ""))
            and str(xml_row.get("dv_selected_tce", "")) == str(dv_row.get("dv_selected_tce", ""))
        )
        row["xml_candidate_match"] = int(bool(same_product and str(xml_row.get("xml_found", "")) in {"1", "1.0"}))
        row["xml_found"] = 0
        row["xml_match_reason"] = "matched" if same_product else ("dv_candidate_mismatch" if not dv_match else "product_mismatch")
        if same_product:
            for key, value in xml_row.items():
                if key.startswith("xml_"):
                    row[key] = value
        out.append(row)
    return out


def fit_transform(rows: list[dict[str, object]], features: list[str]) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    x = np.zeros((len(rows), len(features)), dtype=np.float64)
    stats: dict[str, tuple[float, float]] = {}
    for j, feat in enumerate(features):
        vals = np.array([to_float(row.get(feat)) for row in rows], dtype=np.float64)
        med = np.nanmedian(vals) if np.isfinite(vals).any() else 0.0
        vals = np.where(np.isfinite(vals), vals, med)
        mu = float(vals.mean())
        sig = float(vals.std()) or 1.0
        x[:, j] = (vals - mu) / sig
        stats[feat] = (mu, sig)
    return x, stats


def transform(rows: list[dict[str, object]], features: list[str], stats: dict[str, tuple[float, float]]) -> np.ndarray:
    x = np.zeros((len(rows), len(features)), dtype=np.float64)
    for j, feat in enumerate(features):
        vals = np.array([to_float(row.get(feat)) for row in rows], dtype=np.float64)
        mu, sig = stats[feat]
        vals = np.where(np.isfinite(vals), vals, mu)
        x[:, j] = (vals - mu) / sig
    return x


def split_by_host(rows: list[dict[str, object]], seed: int = 11) -> tuple[list[int], list[int], list[int]]:
    strict = [
        i for i, r in enumerate(rows)
        if r.get("strict_planet_class") in {"known_or_confirmed_planet", "false_positive_or_false_alarm"}
    ]
    hosts = sorted({str(rows[i].get("tic_id") or i) for i in strict})
    random.Random(seed).shuffle(hosts)
    n_test = max(1, int(len(hosts) * 0.25))
    n_cal = max(1, int(len(hosts) * 0.20))
    test_hosts = set(hosts[:n_test])
    cal_hosts = set(hosts[n_test : n_test + n_cal])
    train, cal, test = [], [], []
    for i in strict:
        h = str(rows[i].get("tic_id") or i)
        if h in test_hosts:
            test.append(i)
        elif h in cal_hosts:
            cal.append(i)
        else:
            train.append(i)
    return train, cal, test


def y_for(rows: list[dict[str, object]], idx: list[int]) -> np.ndarray:
    return np.array([1.0 if rows[i].get("strict_planet_class") == "known_or_confirmed_planet" else 0.0 for i in idx])


def train_l2_logistic(x: np.ndarray, y: np.ndarray, l2: float, steps: int = 3000, lr: float = 0.03) -> np.ndarray:
    x1 = np.c_[np.ones(x.shape[0]), x]
    w = np.zeros(x1.shape[1], dtype=np.float64)
    for _ in range(steps):
        z = x1 @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        reg = l2 * w
        reg[0] = 0.0
        grad = x1.T @ (p - y) / max(1, len(y)) + reg
        w -= lr * grad
    return w


def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x1 = np.c_[np.ones(x.shape[0]), x]
    return 1.0 / (1.0 + np.exp(-np.clip(x1 @ w, -40, 40)))


def calibrate_temperature(p: np.ndarray, y: np.ndarray) -> float:
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.5, 5.0, 91):
        pc = 1.0 / (1.0 + np.exp(-logits / t))
        loss = -float(np.mean(y * np.log(np.clip(pc, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pc, 1e-6, 1))))
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return 1.0 / (1.0 + np.exp(-logits / t))


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float | None:
    if len(y) == 0:
        return None
    order = np.argsort(score)[::-1][: min(k, len(y))]
    return float(y[order].mean()) if len(order) else None


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        hi = lo + 1 / bins
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total


def rv_utility(row: dict[str, object]) -> float:
    base = to_float(row.get("score_filtered_followup"))
    if not math.isfinite(base):
        return 0.0
    return max(0.0, base)


def count(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for row in rows:
        c[str(row.get(field, ""))] += 1
    return dict(c)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--dv", default="data/processed/dv_all_final_dedup.csv")
    parser.add_argument("--xml", default="data/processed/dv_xml_all_complete.csv")
    parser.add_argument("--merged", default="data/processed/full_dv_xml_merged.csv")
    parser.add_argument("--report", default="reports/full_triage_metrics.json")
    parser.add_argument("--ranked", default="reports/candidate_triage_ranking.csv")
    parser.add_argument("--period-match-rtol", type=float, default=0.01)
    args = parser.parse_args()

    rows = merge(
        read_csv(pathlib.Path(args.candidates)),
        read_csv(pathlib.Path(args.dv)),
        read_csv(pathlib.Path(args.xml)),
        period_match_rtol=args.period_match_rtol,
    )
    features = [feat for feat in BASE_FEATURES if any(feat in row for row in rows)]
    train_idx, cal_idx, test_idx = split_by_host(rows)
    train_rows = [rows[i] for i in train_idx]
    all_x, stats = fit_transform(train_rows, features)
    y_train = y_for(rows, train_idx)
    w = train_l2_logistic(all_x, y_train, l2=0.01)
    p_cal_raw = predict(transform([rows[i] for i in cal_idx], features, stats), w)
    y_cal = y_for(rows, cal_idx)
    temp = calibrate_temperature(p_cal_raw, y_cal) if len(y_cal) else 1.0
    p_test_raw = predict(transform([rows[i] for i in test_idx], features, stats), w)
    p_test = apply_temperature(p_test_raw, temp)
    y_test = y_for(rows, test_idx)

    report: dict[str, object] = {
        "n_rows": len(rows),
        "class_counts": count(rows, "strict_planet_class"),
        "dv_has_tce_counts": count(rows, "dv_has_tce"),
        "dv_host_has_tce_counts": count(rows, "dv_host_has_tce"),
        "dv_candidate_match_counts": count(rows, "dv_candidate_match"),
        "dv_match_reason_counts": count(rows, "dv_match_reason"),
        "xml_found_counts": count(rows, "xml_found"),
        "xml_host_found_counts": count(rows, "xml_host_found"),
        "xml_candidate_match_counts": count(rows, "xml_candidate_match"),
        "xml_error_counts": count(rows, "xml_error"),
        "n_features": len(features),
        "features": features,
        "n_train": len(train_idx),
        "n_calibration": len(cal_idx),
        "n_test": len(test_idx),
        "temperature": temp,
        "test_brier": brier(y_test, p_test) if len(y_test) else None,
        "test_ece": ece(y_test, p_test) if len(y_test) else None,
        "test_precision_at_k": {str(k): precision_at_k(y_test, p_test, k) for k in (10, 25, 50, 100, 250)},
    }

    x_all = transform(rows, features, stats)
    p_all = apply_temperature(predict(x_all, w), temp)
    for row, prob in zip(rows, p_all):
        row["calibrated_planet_probability"] = float(prob)
        row["rv_utility"] = rv_utility(row)
        row["triage_score"] = float(prob) * rv_utility(row)
    candidates = [
        row for row in rows
        if row.get("strict_planet_class") in {"planet_candidate", "ambiguous_planet_candidate"}
    ]
    candidates.sort(key=lambda r: to_float(r.get("triage_score")), reverse=True)
    write_rows(pathlib.Path(args.merged), rows)
    write_rows(pathlib.Path(args.ranked), candidates[:500])
    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
