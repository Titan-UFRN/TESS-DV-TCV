#!/usr/bin/env python3
"""Build the retrospective tabular TESS candidate-vetting baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
from collections import defaultdict
from typing import Iterable

import numpy as np


NEGATIVE_DISPOSITIONS = {
    "FP",
    "FA",
}

CONFIRMED_PLANET_DISPOSITIONS = {
    "KP",
    "CP",
}

PLANET_CANDIDATE_DISPOSITIONS = {
    "PC",
}

AMBIGUOUS_CANDIDATE_DISPOSITIONS = {
    "APC",
}


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str], key: str) -> float:
    value = (row.get(key) or "").strip()
    if not value:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def first_present(row: dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def safe_log10(x: float) -> float:
    if not math.isfinite(x) or x <= 0:
        return math.nan
    return math.log10(x)


def expected_k_ms(period_days: float, radius_rearth: float, stellar_mass_msun: float) -> float:
    """Approximate RV semi-amplitude using a rough radius-mass relation.

    This is a baseline ranking feature, not a physical inference result.
    Assumes circular orbit and sin(i)=1.
    """
    if not all(math.isfinite(v) and v > 0 for v in (period_days, radius_rearth, stellar_mass_msun)):
        return math.nan
    if radius_rearth < 1.5:
        mass_mearth = radius_rearth**3.7
    elif radius_rearth < 4.0:
        mass_mearth = 2.7 * radius_rearth**1.3
    else:
        mass_mearth = 10.0 * radius_rearth
    return 0.0895 * mass_mearth * (period_days / 365.25) ** (-1.0 / 3.0) * stellar_mass_msun ** (-2.0 / 3.0)


def normalize_tic(value: str) -> str:
    return "".join(ch for ch in (value or "") if ch.isdigit())


def build_mass_lookup(rows: list[dict[str, str]], id_field: str = "tic_id") -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        tic = normalize_tic(row.get(id_field, ""))
        if not tic:
            continue
        existing = lookup.get(tic)
        if existing is None or (not existing.get("st_mass") and row.get("st_mass")):
            lookup[tic] = row
    return lookup


def stellar_mass_proxy(stellar_radius: float) -> tuple[float, str]:
    """Fallback for first-pass ranking when TOI lacks stellar mass.

    This proxy is intentionally simple and must be replaced by TIC/Gaia/literature
    masses before any scientific claim about RV feasibility.
    """
    if math.isfinite(stellar_radius) and 0.08 <= stellar_radius <= 3.0:
        return stellar_radius, "radius_proxy"
    return math.nan, "missing"


def build_candidates(
    toi_rows: list[dict[str, str]],
    mass_lookup: dict[str, dict[str, str]],
    tic_lookup: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for row in toi_rows:
        disp = first_present(row, ["tfopwg_disp", "tsv_disposition"]).upper()
        period = to_float(row, "pl_orbper")
        radius = to_float(row, "pl_rade")
        tic_id = normalize_tic(first_present(row, ["tid"]))
        archive_row = mass_lookup.get(tic_id, {})
        tic_row = tic_lookup.get(tic_id, {})
        stellar_mass = to_float(tic_row, "mass")
        stellar_radius = to_float(tic_row, "rad")
        teff = to_float(tic_row, "Teff")
        logg = to_float(tic_row, "logg")
        tmag = to_float(tic_row, "Tmag")
        stellar_mass_source = "tic" if math.isfinite(stellar_mass) else "missing"
        stellar_radius_source = "tic" if math.isfinite(stellar_radius) else "missing"
        if not math.isfinite(stellar_mass):
            stellar_mass = to_float(archive_row, "st_mass")
            stellar_mass_source = "ps_or_pscomppars" if math.isfinite(stellar_mass) else "missing"
        if not math.isfinite(stellar_radius):
            stellar_radius = to_float(row, "st_rad")
            stellar_radius_source = "toi" if math.isfinite(stellar_radius) else "missing"
        if not math.isfinite(stellar_mass):
            stellar_mass, stellar_mass_source = stellar_mass_proxy(stellar_radius)
        if not math.isfinite(tmag):
            tmag = to_float(row, "st_tmag")
        depth = to_float(row, "pl_trandep")
        duration = to_float(row, "pl_trandurh")
        if not math.isfinite(teff):
            teff = to_float(row, "st_teff")
        if not math.isfinite(logg):
            logg = to_float(row, "st_logg")
        expected_k = expected_k_ms(period, radius, stellar_mass)
        flags = binary_like_flags(radius, depth, period, duration)
        binary_risk = binary_like_risk(flags)

        strict_class = strict_planet_class(disp)
        supervised_label = supervised_planet_label(strict_class)
        broad_label = broad_planet_label(strict_class)
        rv_label = rv_followup_label(strict_class, expected_k, tmag, radius)

        candidates.append(
            {
                "tic_id": tic_id,
                "toi": first_present(row, ["toi"]),
                "tfopwg_disp": disp,
                "archive_pl_name": first_present(archive_row, ["pl_name"]),
                "archive_hostname": first_present(archive_row, ["hostname"]),
                "archive_rv_flag": first_present(archive_row, ["rv_flag"]),
                "stellar_mass_source": stellar_mass_source,
                "stellar_radius_source": stellar_radius_source,
                "tic_contratio": to_float(tic_row, "contratio"),
                "tic_numcont": to_float(tic_row, "numcont"),
                "label": supervised_label,
                "label_name": strict_class,
                "strict_planet_class": strict_class,
                "broad_planet_label": broad_label,
                "rv_followup_label": rv_label,
                "period_days": period,
                "planet_radius_rearth": radius,
                "stellar_mass_msun": stellar_mass,
                "stellar_radius_rsun": stellar_radius,
                "tmag": tmag,
                "transit_depth_ppm_or_percent_raw": depth,
                "transit_duration_hours": duration,
                "stellar_teff": teff,
                "stellar_logg": logg,
                "expected_k_ms": expected_k,
                "score_expected_k": expected_k if math.isfinite(expected_k) else math.nan,
                "score_simple_followup": simple_followup_score(expected_k, tmag, radius, period),
                "flag_large_radius": flags["large_radius"],
                "flag_giant_radius": flags["giant_radius"],
                "flag_deep_transit": flags["deep_transit"],
                "flag_extreme_depth": flags["extreme_depth"],
                "flag_ultra_short_period": flags["ultra_short_period"],
                "flag_short_period": flags["short_period"],
                "flag_long_duration_short_period": flags["long_duration_short_period"],
                "binary_like_risk": binary_risk,
                "score_filtered_followup": filtered_followup_score(expected_k, tmag, radius, period, binary_risk),
            }
        )
    return candidates


def binary_like_flags(radius: float, depth: float, period: float, duration: float) -> dict[str, int]:
    return {
        "large_radius": int(math.isfinite(radius) and radius >= 8.0),
        "giant_radius": int(math.isfinite(radius) and radius >= 15.0),
        "deep_transit": int(math.isfinite(depth) and depth >= 20000.0),
        "extreme_depth": int(math.isfinite(depth) and depth >= 50000.0),
        "ultra_short_period": int(math.isfinite(period) and period < 2.0),
        "short_period": int(math.isfinite(period) and period < 5.0),
        "long_duration_short_period": int(
            math.isfinite(duration)
            and math.isfinite(period)
            and period > 0
            and (duration / (period * 24.0)) >= 0.12
        ),
    }


def binary_like_risk(flags: dict[str, int]) -> float:
    score = (
        0.35 * flags["large_radius"]
        + 0.35 * flags["giant_radius"]
        + 0.25 * flags["deep_transit"]
        + 0.25 * flags["extreme_depth"]
        + 0.10 * flags["ultra_short_period"]
        + 0.10 * flags["long_duration_short_period"]
    )
    return min(1.0, score)


def strict_planet_class(disposition: str) -> str:
    if disposition in CONFIRMED_PLANET_DISPOSITIONS:
        return "known_or_confirmed_planet"
    if disposition in PLANET_CANDIDATE_DISPOSITIONS:
        return "planet_candidate"
    if disposition in AMBIGUOUS_CANDIDATE_DISPOSITIONS:
        return "ambiguous_planet_candidate"
    if disposition in NEGATIVE_DISPOSITIONS:
        return "false_positive_or_false_alarm"
    return "unknown"


def supervised_planet_label(strict_class: str) -> int | str:
    if strict_class == "known_or_confirmed_planet":
        return 1
    if strict_class == "false_positive_or_false_alarm":
        return 0
    return ""


def broad_planet_label(strict_class: str) -> str:
    if strict_class == "known_or_confirmed_planet":
        return "confirmed_positive"
    if strict_class in {"planet_candidate", "ambiguous_planet_candidate"}:
        return "candidate_unlabeled_for_strict_metrics"
    if strict_class == "false_positive_or_false_alarm":
        return "negative"
    return "unknown"


def rv_followup_label(strict_class: str, expected_k: float, tmag: float, radius: float) -> str:
    if strict_class == "false_positive_or_false_alarm":
        return "not_planet"
    if strict_class == "unknown":
        return "unknown"
    if not math.isfinite(expected_k):
        return "missing_rv_feasibility"
    bright_enough = (not math.isfinite(tmag)) or tmag <= 12.0
    small_or_subneptune = math.isfinite(radius) and radius <= 4.0
    if expected_k >= 2.0 and bright_enough:
        return "rv_high_value"
    if expected_k >= 1.0 and bright_enough and small_or_subneptune:
        return "rv_science_interesting"
    if expected_k < 1.0 or not bright_enough:
        return "rv_low_practicality"
    return "rv_medium_value"


def simple_followup_score(expected_k: float, tmag: float, radius: float, period: float) -> float:
    if not math.isfinite(expected_k):
        return math.nan
    mag_penalty = 1.0
    if math.isfinite(tmag):
        mag_penalty = 10 ** (-0.2 * max(0.0, tmag - 10.0))
    small_planet_weight = 1.0
    if math.isfinite(radius) and radius <= 4.0:
        small_planet_weight = 1.5
    period_penalty = 1.0
    if math.isfinite(period) and period > 50:
        period_penalty = math.sqrt(50.0 / period)
    return expected_k * mag_penalty * small_planet_weight * period_penalty


def filtered_followup_score(expected_k: float, tmag: float, radius: float, period: float, binary_risk: float) -> float:
    base = simple_followup_score(expected_k, tmag, radius, period)
    if not math.isfinite(base):
        return math.nan
    return base * (1.0 - 0.85 * binary_risk)


def write_rows(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_false_positive_audit(path: pathlib.Path, rows: list[dict[str, object]], score_name: str, top_k: int = 250) -> dict[str, object]:
    ranked = sorted(
        rows,
        key=lambda r: numeric_score(r, score_name),
        reverse=True,
    )
    top = ranked[:top_k]
    fps = [r for r in top if r["strict_planet_class"] == "false_positive_or_false_alarm"]
    audit_rows = []
    for rank, row in enumerate(ranked, start=1):
        if rank > top_k:
            break
        if row["strict_planet_class"] != "false_positive_or_false_alarm":
            continue
        radius = row.get("planet_radius_rearth")
        period = row.get("period_days")
        reasons = []
        if isinstance(radius, float) and math.isfinite(radius) and radius >= 8.0:
            reasons.append("large_radius_binary_like")
        if isinstance(period, float) and math.isfinite(period) and period < 2.0:
            reasons.append("ultra_short_period_high_k")
        if row.get("stellar_mass_source") == "radius_proxy":
            reasons.append("stellar_mass_proxy_used")
        if numeric_score(row, "binary_like_risk") >= 0.5:
            reasons.append("high_binary_like_risk")
        if row.get("archive_rv_flag") == "1":
            reasons.append("archive_rv_flag_present")
        audit = dict(row)
        audit["rank"] = rank
        audit["ranking_score"] = numeric_score(row, score_name)
        audit["audit_reasons"] = ";".join(reasons)
        audit_rows.append(audit)
    write_rows(path, audit_rows)
    return {
        "score": score_name,
        "top_k": top_k,
        "false_positives_in_top_k": len(fps),
        "false_positive_fraction_top_k": len(fps) / top_k if top_k else None,
        "audit_path": str(path),
    }


def numeric_score(row: dict[str, object], score_name: str) -> float:
    value = row.get(score_name)
    if isinstance(value, float) and math.isfinite(value):
        return value
    try:
        f = float(value)  # type: ignore[arg-type]
        return f if math.isfinite(f) else -math.inf
    except (TypeError, ValueError):
        return -math.inf


def feature_matrix(rows: list[dict[str, object]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labeled = [r for r in rows if r["label"] in (0, 1)]
    y = np.array([int(r["label"]) for r in labeled], dtype=np.float64)
    host_keys = [str(r.get("tic_id") or r.get("toi") or i) for i, r in enumerate(labeled)]
    x = np.zeros((len(labeled), len(feature_names)), dtype=np.float64)
    for j, name in enumerate(feature_names):
        values = np.array([float(r[name]) if isinstance(r[name], float) and math.isfinite(float(r[name])) else math.nan for r in labeled])
        median = np.nanmedian(values) if np.isfinite(values).any() else 0.0
        values = np.where(np.isfinite(values), values, median)
        x[:, j] = values
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (x - mu) / sigma, y, host_keys


def split_by_host(host_keys: list[str], seed: int = 42, test_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    unique = sorted(set(host_keys))
    rng.shuffle(unique)
    test_hosts = set(unique[: max(1, int(len(unique) * test_frac))])
    test = np.array([h in test_hosts for h in host_keys], dtype=bool)
    return ~test, test


def train_logistic(x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, steps: int = 2500, lr: float = 0.05) -> np.ndarray:
    x1 = np.c_[np.ones(x.shape[0]), x]
    w = np.zeros(x1.shape[1], dtype=np.float64)
    xt = x1[train_mask]
    yt = y[train_mask]
    for _ in range(steps):
        z = xt @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        grad = xt.T @ (p - yt) / len(yt)
        w -= lr * grad
    return w


def predict_logistic(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x1 = np.c_[np.ones(x.shape[0]), x]
    z = x1 @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def precision_at_k(y: np.ndarray, score: np.ndarray, mask: np.ndarray, k: int) -> float | None:
    idx = np.where(mask & np.isfinite(score))[0]
    if len(idx) == 0:
        return None
    order = idx[np.argsort(score[idx])[::-1]]
    top = order[: min(k, len(order))]
    return float(y[top].mean()) if len(top) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()

    raw = pathlib.Path(args.raw)
    out = pathlib.Path(args.out)
    reports = pathlib.Path(args.reports)
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    toi = read_csv(raw / "toi.csv")
    ps = read_csv(raw / "ps.csv")
    pscomppars = read_csv(raw / "pscomppars.csv")
    tic_path = raw / "tic_stellar.csv"
    tic_rows = read_csv(tic_path) if tic_path.exists() else []
    mass_lookup = build_mass_lookup(pscomppars)
    mass_lookup.update(build_mass_lookup(ps))
    tic_lookup = build_mass_lookup(tic_rows, id_field="ID")
    candidates = build_candidates(toi, mass_lookup, tic_lookup)
    write_rows(out / "candidates.csv", candidates)

    feature_names = [
        "period_days",
        "planet_radius_rearth",
        "stellar_mass_msun",
        "stellar_radius_rsun",
        "tmag",
        "transit_duration_hours",
        "stellar_teff",
        "stellar_logg",
        "expected_k_ms",
        "score_simple_followup",
        "binary_like_risk",
        "score_filtered_followup",
    ]
    x, y, host_keys = feature_matrix(candidates, feature_names)
    train_mask, test_mask = split_by_host(host_keys)
    metrics: dict[str, object] = {
        "n_candidates": len(candidates),
        "n_labeled": int(len(y)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "positive_rate_labeled": float(y.mean()) if len(y) else None,
        "label_counts": dict(label_counts(candidates)),
        "strict_planet_class_counts": dict(count_field(candidates, "strict_planet_class")),
        "broad_planet_label_counts": dict(count_field(candidates, "broad_planet_label")),
        "rv_followup_label_counts": dict(count_field(candidates, "rv_followup_label")),
        "stellar_mass_source_counts": dict(count_field(candidates, "stellar_mass_source")),
        "stellar_radius_source_counts": dict(count_field(candidates, "stellar_radius_source")),
        "features": feature_names,
        "warning": "Strict supervised metrics use only KP/CP versus FP/FA. PC/APC/unknown are excluded, not treated as false positives.",
    }

    if len(y) >= 20 and train_mask.sum() >= 10 and test_mask.sum() >= 5 and len(set(y.astype(int))) == 2:
        w = train_logistic(x, y, train_mask)
        p = predict_logistic(x, w)
        metrics["logistic_precision_at_k"] = {str(k): precision_at_k(y, p, test_mask, k) for k in (10, 25, 50, 100)}
        metrics["expected_k_precision_at_k"] = precision_from_candidate_feature(candidates, y, test_mask, "score_expected_k")
        metrics["simple_followup_precision_at_k"] = precision_from_candidate_feature(candidates, y, test_mask, "score_simple_followup")
        metrics["filtered_followup_precision_at_k"] = precision_from_candidate_feature(candidates, y, test_mask, "score_filtered_followup")
    else:
        metrics["model_status"] = "not enough labeled data or only one class"

    metrics["false_positive_audit"] = {
        "expected_k": write_false_positive_audit(reports / "false_positive_audit_expected_k.csv", candidates, "score_expected_k"),
        "simple_followup": write_false_positive_audit(reports / "false_positive_audit_simple_followup.csv", candidates, "score_simple_followup"),
        "filtered_followup": write_false_positive_audit(reports / "false_positive_audit_filtered_followup.csv", candidates, "score_filtered_followup"),
    }
    ranked = sorted(candidates, key=lambda r: numeric_score(r, "score_simple_followup"), reverse=True)
    write_rows(reports / "top_ranked_candidates.csv", ranked[:250])
    (reports / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def label_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["label_name"])] += 1
    return counts


def count_field(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(field) or "")] += 1
    return counts


def precision_from_candidate_feature(rows: list[dict[str, object]], y: np.ndarray, mask: np.ndarray, feature: str) -> dict[str, float | None]:
    labeled = [r for r in rows if r["label"] in (0, 1)]
    score = np.array([
        float(r[feature]) if isinstance(r[feature], float) and math.isfinite(float(r[feature])) else math.nan
        for r in labeled
    ])
    return {str(k): precision_at_k(y, score, mask, k) for k in (10, 25, 50, 100)}


if __name__ == "__main__":
    raise SystemExit(main())
