#!/usr/bin/env python3
"""Evaluate strict-label baselines on the small DV metadata subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
from collections import defaultdict

import numpy as np


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


def merge(candidates: list[dict[str, str]], dv_rows: list[dict[str, str]], xml_rows: list[dict[str, str]] | None = None) -> list[dict[str, object]]:
    by_tic = {row["tic_id"]: row for row in candidates}
    xml_by_tic = {row["tic_id"]: row for row in (xml_rows or [])}
    merged: list[dict[str, object]] = []
    for dv in dv_rows:
        row = dict(by_tic.get(dv["tic_id"], {}))
        row.update({f"dv_{k}" if k in row else k: v for k, v in dv.items()})
        xml = xml_by_tic.get(dv["tic_id"], {})
        row.update({f"xml_{k}" if k in row else k: v for k, v in xml.items()})
        merged.append(row)
    return merged


def numeric_features(rows: list[dict[str, object]]) -> list[str]:
    base = [
        "period_days",
        "planet_radius_rearth",
        "stellar_mass_msun",
        "stellar_radius_rsun",
        "tmag",
        "transit_depth_ppm_or_percent_raw",
        "transit_duration_hours",
        "stellar_teff",
        "stellar_logg",
        "expected_k_ms",
        "binary_like_risk",
        "score_filtered_followup",
        "dv_has_tce",
        "dv_tce_count",
    ]
    dv_extra = []
    for row in rows:
        for key, value in row.items():
            if (key.startswith("dv_") or key.startswith("xml_")) and key not in base and math.isfinite(to_float(value)):
                dv_extra.append(key)
    return base + sorted(set(dv_extra))


def labeled_xy(rows: list[dict[str, object]], features: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labeled = [
        r for r in rows
        if r.get("strict_planet_class") in {"known_or_confirmed_planet", "false_positive_or_false_alarm"}
    ]
    y = np.array([1.0 if r.get("strict_planet_class") == "known_or_confirmed_planet" else 0.0 for r in labeled])
    host = [str(r.get("tic_id") or i) for i, r in enumerate(labeled)]
    x = np.zeros((len(labeled), len(features)), dtype=np.float64)
    for j, feat in enumerate(features):
        vals = np.array([to_float(r.get(feat)) for r in labeled], dtype=np.float64)
        med = np.nanmedian(vals) if np.isfinite(vals).any() else 0.0
        vals = np.where(np.isfinite(vals), vals, med)
        x[:, j] = vals
    mu = x.mean(axis=0)
    sig = x.std(axis=0)
    sig[sig == 0] = 1.0
    return (x - mu) / sig, y, host


def split(host: list[str], seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    uniq = sorted(set(host))
    rng.shuffle(uniq)
    test_hosts = set(uniq[: max(1, int(len(uniq) * 0.35))])
    test = np.array([h in test_hosts for h in host], dtype=bool)
    return ~test, test


def train_logistic(x: np.ndarray, y: np.ndarray, train: np.ndarray, steps: int = 2000, lr: float = 0.05) -> np.ndarray:
    x1 = np.c_[np.ones(x.shape[0]), x]
    w = np.zeros(x1.shape[1], dtype=np.float64)
    for _ in range(steps):
        z = x1[train] @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        grad = x1[train].T @ (p - y[train]) / max(1, train.sum())
        w -= lr * grad
    return w


def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x1 = np.c_[np.ones(x.shape[0]), x]
    return 1.0 / (1.0 + np.exp(-np.clip(x1 @ w, -40, 40)))


def precision_at_k(y: np.ndarray, score: np.ndarray, mask: np.ndarray, k: int) -> float | None:
    idx = np.where(mask & np.isfinite(score))[0]
    if len(idx) == 0:
        return None
    order = idx[np.argsort(score[idx])[::-1]][: min(k, len(idx))]
    return float(y[order].mean()) if len(order) else None


def score_array(rows: list[dict[str, object]], feature: str) -> np.ndarray:
    labeled = [
        r for r in rows
        if r.get("strict_planet_class") in {"known_or_confirmed_planet", "false_positive_or_false_alarm"}
    ]
    return np.array([to_float(r.get(feature)) for r in labeled], dtype=np.float64)


def count(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for row in rows:
        c[str(row.get(field, ""))] += 1
    return dict(c)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--dv", default="data/processed/dv_subset.csv")
    parser.add_argument("--xml", default="")
    parser.add_argument("--out", default="data/processed/dv_merged_subset.csv")
    parser.add_argument("--report", default="reports/dv_subset_metrics.json")
    args = parser.parse_args()

    xml_rows = read_csv(pathlib.Path(args.xml)) if args.xml else None
    rows = merge(read_csv(pathlib.Path(args.candidates)), read_csv(pathlib.Path(args.dv)), xml_rows)
    features = numeric_features(rows)
    x, y, host = labeled_xy(rows, features)
    train, test = split(host)
    metrics: dict[str, object] = {
        "n_subset": len(rows),
        "n_labeled": int(len(y)),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "class_counts": count(rows, "strict_planet_class"),
        "dv_group_counts": count(rows, "dv_subset_group"),
        "dv_has_tce_counts": count(rows, "dv_has_tce"),
        "xml_found_counts": count(rows, "xml_found"),
        "xml_error_counts": count(rows, "xml_error"),
        "n_features": len(features),
        "features": features,
    }
    if len(y) >= 20 and len(set(y.astype(int))) == 2:
        w = train_logistic(x, y, train)
        p = predict(x, w)
        metrics["dv_logistic_precision_at_k"] = {str(k): precision_at_k(y, p, test, k) for k in (5, 10, 25, 50)}
        metrics["filtered_followup_precision_at_k"] = {
            str(k): precision_at_k(y, score_array(rows, "score_filtered_followup"), test, k)
            for k in (5, 10, 25, 50)
        }
    write_rows(pathlib.Path(args.out), rows)
    pathlib.Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.report).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
