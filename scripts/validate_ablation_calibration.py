#!/usr/bin/env python3
"""Run modality ablations, reliability curves, and subgroup calibration metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
from collections import defaultdict

import numpy as np


FEATURE_SETS = {
    "archive_tic": [
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
    ],
    "archive_tic_dv_header": [
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
    ],
    "archive_tic_xml": [
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
    ],
}
FEATURE_SETS["all_features"] = sorted(set(FEATURE_SETS["archive_tic_dv_header"] + FEATURE_SETS["archive_tic_xml"]))


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


def strict_indices(rows: list[dict[str, str]]) -> list[int]:
    return [
        i for i, row in enumerate(rows)
        if row.get("strict_planet_class") in {"known_or_confirmed_planet", "false_positive_or_false_alarm"}
    ]


def y_for(rows: list[dict[str, str]], idx: list[int]) -> np.ndarray:
    return np.array([1.0 if rows[i].get("strict_planet_class") == "known_or_confirmed_planet" else 0.0 for i in idx])


def split_indices(rows: list[dict[str, str]], seed: int = 11) -> tuple[list[int], list[int], list[int]]:
    idx = strict_indices(rows)
    hosts = sorted({str(rows[i].get("tic_id") or i) for i in idx})
    random.Random(seed).shuffle(hosts)
    n_test = max(1, int(len(hosts) * 0.25))
    n_cal = max(1, int(len(hosts) * 0.20))
    test_hosts = set(hosts[:n_test])
    cal_hosts = set(hosts[n_test : n_test + n_cal])
    train, cal, test = [], [], []
    for i in idx:
        host = str(rows[i].get("tic_id") or i)
        if host in test_hosts:
            test.append(i)
        elif host in cal_hosts:
            cal.append(i)
        else:
            train.append(i)
    return train, cal, test


def available_features(rows: list[dict[str, str]], features: list[str]) -> list[str]:
    return [feat for feat in features if any(feat in row for row in rows)]


def fit_transform(rows: list[dict[str, str]], idx: list[int], features: list[str]) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    x = np.zeros((len(idx), len(features)), dtype=np.float64)
    stats: dict[str, tuple[float, float]] = {}
    for j, feat in enumerate(features):
        vals = np.array([to_float(rows[i].get(feat)) for i in idx], dtype=np.float64)
        med = np.nanmedian(vals) if np.isfinite(vals).any() else 0.0
        vals = np.where(np.isfinite(vals), vals, med)
        mu = float(vals.mean())
        sig = float(vals.std()) or 1.0
        x[:, j] = (vals - mu) / sig
        stats[feat] = (mu, sig)
    return x, stats


def transform(rows: list[dict[str, str]], idx: list[int], features: list[str], stats: dict[str, tuple[float, float]]) -> np.ndarray:
    x = np.zeros((len(idx), len(features)), dtype=np.float64)
    for j, feat in enumerate(features):
        mu, sig = stats[feat]
        vals = np.array([to_float(rows[i].get(feat)) for i in idx], dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, mu)
        x[:, j] = (vals - mu) / sig
    return x


def train_l2(x: np.ndarray, y: np.ndarray, l2: float = 0.01, steps: int = 3000, lr: float = 0.03) -> np.ndarray:
    x1 = np.c_[np.ones(x.shape[0]), x]
    w = np.zeros(x1.shape[1], dtype=np.float64)
    for _ in range(steps):
        z = x1 @ w
        p = sigmoid(z)
        reg = l2 * w
        reg[0] = 0.0
        grad = x1.T @ (p - y) / max(1, len(y)) + reg
        w -= lr * grad
    return w


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return sigmoid(np.c_[np.ones(x.shape[0]), x] @ w)


def calibrate_temperature(p: np.ndarray, y: np.ndarray) -> float:
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.5, 5.0, 91):
        pc = sigmoid(logits / t)
        loss = -float(np.mean(y * np.log(np.clip(pc, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pc, 1e-6, 1))))
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return sigmoid(logits / t)


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float | None:
    if len(y) == 0:
        return None
    order = np.argsort(score)[::-1][: min(k, len(y))]
    return float(y[order].mean()) if len(order) else None


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    total = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        mask = (p >= lo) & (p <= hi if b == bins - 1 else p < hi)
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total


def reliability_rows(model: str, y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, object]]:
    out = []
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        mask = (p >= lo) & (p <= hi if b == bins - 1 else p < hi)
        out.append({
            "model": model,
            "bin": b,
            "bin_min": lo,
            "bin_max": hi,
            "n": int(mask.sum()),
            "mean_probability": float(p[mask].mean()) if mask.any() else "",
            "empirical_positive_rate": float(y[mask].mean()) if mask.any() else "",
        })
    return out


def subgroup_name(row: dict[str, str]) -> list[str]:
    groups = []
    groups.append("has_xml" if row.get("xml_found") in {"1", "1.0"} else "no_xml")
    radius = to_float(row.get("planet_radius_rearth"))
    if math.isfinite(radius):
        if radius < 2:
            groups.append("radius_lt2")
        elif radius < 4:
            groups.append("radius_2_4")
        elif radius < 8:
            groups.append("radius_4_8")
        else:
            groups.append("radius_ge8")
    period = to_float(row.get("period_days"))
    if math.isfinite(period):
        groups.append("period_lt5" if period < 5 else "period_ge5")
    tmag = to_float(row.get("tmag"))
    if math.isfinite(tmag):
        groups.append("tmag_lt10" if tmag < 10 else "tmag_ge10")
    return groups


def subgroup_metrics(rows: list[dict[str, str]], idx: list[int], y: np.ndarray, p: np.ndarray, model: str) -> list[dict[str, object]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for local_i, row_i in enumerate(idx):
        for group in subgroup_name(rows[row_i]):
            buckets[group].append(local_i)
    out = []
    for group, local_idx in sorted(buckets.items()):
        mask = np.array(local_idx, dtype=int)
        if len(mask) < 10:
            continue
        yg = y[mask]
        pg = p[mask]
        out.append({
            "model": model,
            "subgroup": group,
            "n": int(len(mask)),
            "positive_rate": float(yg.mean()),
            "brier": brier(yg, pg),
            "ece": ece(yg, pg),
        })
    return out


def evaluate_model(rows: list[dict[str, str]], train_idx: list[int], cal_idx: list[int], test_idx: list[int], name: str, feature_names: list[str]) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    features = available_features(rows, feature_names)
    x_train, stats = fit_transform(rows, train_idx, features)
    y_train = y_for(rows, train_idx)
    w = train_l2(x_train, y_train)
    p_cal_raw = predict(transform(rows, cal_idx, features, stats), w)
    y_cal = y_for(rows, cal_idx)
    temp = calibrate_temperature(p_cal_raw, y_cal)
    p_test_raw = predict(transform(rows, test_idx, features, stats), w)
    p_test = apply_temperature(p_test_raw, temp)
    y_test = y_for(rows, test_idx)
    metrics = {
        "model": name,
        "n_features": len(features),
        "temperature": temp,
        "brier": brier(y_test, p_test),
        "ece": ece(y_test, p_test),
        "precision_at_10": precision_at_k(y_test, p_test, 10),
        "precision_at_25": precision_at_k(y_test, p_test, 25),
        "precision_at_50": precision_at_k(y_test, p_test, 50),
        "precision_at_100": precision_at_k(y_test, p_test, 100),
        "precision_at_250": precision_at_k(y_test, p_test, 250),
    }
    return metrics, reliability_rows(name, y_test, p_test), subgroup_metrics(rows, test_idx, y_test, p_test, name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", default="data/processed/full_dv_xml_merged.csv")
    parser.add_argument("--outdir", default="reports/validation")
    args = parser.parse_args()

    rows = read_csv(pathlib.Path(args.merged))
    train_idx, cal_idx, test_idx = split_indices(rows)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = []
    reliability = []
    subgroups = []
    for name, features in FEATURE_SETS.items():
        m, rel, sub = evaluate_model(rows, train_idx, cal_idx, test_idx, name, features)
        metrics.append(m)
        reliability.extend(rel)
        subgroups.extend(sub)
        print(name, json.dumps(m, sort_keys=True), flush=True)

    write_rows(outdir / "ablation_metrics.csv", metrics)
    write_rows(outdir / "reliability_curve.csv", reliability)
    write_rows(outdir / "subgroup_calibration.csv", subgroups)
    (outdir / "ablation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
