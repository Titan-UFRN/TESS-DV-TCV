#!/usr/bin/env python3
"""Reviewer-requested temporal sensitivity and event-association analyses.

This module is intentionally NumPy-only.  It treats the temporal endpoint as a
later TOI disposition, not independent planet truth or realized RV yield.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
from collections import Counter, defaultdict

import numpy as np

from validate_temporal import (
    DV_FEATURES,
    SNAPSHOT_FEATURES,
    XML_FEATURES,
    average_precision,
    ece,
    labels,
    merge,
    metrics,
    read_csv,
    roc_auc,
    write_rows,
)


KS = (10, 25, 50, 100, 250)
LOWER_IS_BETTER = {"brier", "ece", "nll", "error_rate"}


def number(value: object) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
        return result if math.isfinite(result) else math.nan
    except (TypeError, ValueError):
        return math.nan


def phase_error(epoch_a: float, epoch_b: float, period: float) -> float:
    return abs((epoch_a - epoch_b + 0.5 * period) % period - 0.5 * period)


def selected_tce_number(value: str) -> int | None:
    try:
        return int(value.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return None


def annotate_event_association(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source in rows:
        row: dict[str, object] = dict(source)
        period = number(row.get("historic_period_days"))
        epoch = number(row.get("historic_epoch_btjd"))
        duration_days = number(row.get("historic_transit_duration_hours")) / 24.0
        dv_period = number(row.get("dv_data_tperiod"))
        dv_epoch = number(row.get("dv_data_tepoch"))
        xml_period = number(row.get("xml_planetcandidate_orbitalperiodindays"))
        xml_epoch = number(row.get("xml_planetcandidate_epochtjd"))
        toi_value = number(row.get("toi"))
        xml_toi = number(row.get("xml_planetresults_toiid"))
        tce = selected_tce_number(str(row.get("dv_selected_tce", "")))
        xml_planet = number(row.get("xml_planetcandidate_planetnumber"))

        period_tol = 0.01
        epoch_tol = max(0.02, 0.25 * duration_days) if math.isfinite(duration_days) else 0.02
        dv_period_error = abs(dv_period - period) / period if period > 0 and math.isfinite(dv_period) else math.nan
        dv_epoch_error = phase_error(dv_epoch, epoch, period) if period > 0 and math.isfinite(dv_epoch) and math.isfinite(epoch) else math.nan
        xml_period_error = abs(xml_period - period) / period if period > 0 and math.isfinite(xml_period) else math.nan
        xml_epoch_error = phase_error(xml_epoch, epoch, period) if period > 0 and math.isfinite(xml_epoch) and math.isfinite(epoch) else math.nan

        dv_strict = (
            str(row.get("dv_candidate_match", "0")) in {"1", "1.0"}
            and math.isfinite(dv_period_error)
            and dv_period_error <= period_tol
            and math.isfinite(dv_epoch_error)
            and dv_epoch_error <= epoch_tol
            and tce is not None
        )
        xml_strict = (
            dv_strict
            and str(row.get("xml_found", "0")) in {"1", "1.0"}
            and math.isfinite(xml_period_error)
            and xml_period_error <= period_tol
            and math.isfinite(xml_epoch_error)
            and xml_epoch_error <= epoch_tol
            and math.isfinite(xml_planet)
            and int(xml_planet) == tce
            and (not math.isfinite(xml_toi) or abs(xml_toi - toi_value) < 0.001)
        )
        reasons = []
        if str(row.get("dv_candidate_match", "0")) in {"1", "1.0"} and not dv_strict:
            if not math.isfinite(dv_epoch_error) or dv_epoch_error > epoch_tol:
                reasons.append("dv_epoch")
            if not math.isfinite(dv_period_error) or dv_period_error > period_tol:
                reasons.append("dv_period")
            if tce is None:
                reasons.append("missing_tce")
        if str(row.get("xml_found", "0")) in {"1", "1.0"} and not xml_strict:
            if not dv_strict:
                reasons.append("dv_not_strict")
            if not math.isfinite(xml_period_error) or xml_period_error > period_tol:
                reasons.append("xml_period")
            if not math.isfinite(xml_epoch_error) or xml_epoch_error > epoch_tol:
                reasons.append("xml_epoch")
            if not math.isfinite(xml_planet) or int(xml_planet) != tce:
                reasons.append("xml_tce")
            if math.isfinite(xml_toi) and abs(xml_toi - toi_value) >= 0.001:
                reasons.append("xml_toi")

        row.update(
            {
                "event_dv_strict": int(dv_strict),
                "event_xml_strict": int(xml_strict),
                "event_ambiguous": int(
                    str(row.get("dv_candidate_match", "0")) in {"1", "1.0"} and not xml_strict
                ),
                "event_ambiguity_reason": ";".join(sorted(set(reasons))),
                "event_dv_period_rel_error": dv_period_error,
                "event_dv_epoch_phase_error_days": dv_epoch_error,
                "event_xml_period_rel_error": xml_period_error,
                "event_xml_epoch_phase_error_days": xml_epoch_error,
                "event_epoch_tolerance_days": epoch_tol,
            }
        )
        # Strict features never carry a DVR record across an unverified TCE.
        row["dv_has_tce_strict"] = int(dv_strict)
        row["xml_found_strict"] = int(xml_strict)
        for field in DV_FEATURES:
            if field != "dv_has_tce":
                row[field + "_strict"] = row.get(field, "") if dv_strict else ""
        for field in XML_FEATURES:
            if field != "xml_found":
                row[field + "_strict"] = row.get(field, "") if xml_strict else ""
        output.append(row)
    return output


DV_STRICT_FEATURES = ["dv_has_tce_strict"] + [field + "_strict" for field in DV_FEATURES if field != "dv_has_tce"]
XML_STRICT_FEATURES = ["xml_found_strict"] + [field + "_strict" for field in XML_FEATURES if field != "xml_found"]
FEATURE_SETS = {
    "catalog": SNAPSHOT_FEATURES,
    "catalog_availability": SNAPSHOT_FEATURES + ["dv_has_tce_strict", "xml_found_strict"],
    "catalog_dv_strict": SNAPSHOT_FEATURES + DV_STRICT_FEATURES,
    "catalog_xml_strict": SNAPSHOT_FEATURES + XML_STRICT_FEATURES,
    "all_strict": SNAPSHOT_FEATURES + DV_STRICT_FEATURES + XML_STRICT_FEATURES,
}


def design(
    rows: list[dict[str, object]], indices: list[int], features: list[str],
    prep: dict[str, tuple[float, float, float]] | None = None,
    add_missing: bool = False,
) -> tuple[np.ndarray, dict[str, tuple[float, float, float]]]:
    fitted: dict[str, tuple[float, float, float]] = {} if prep is None else prep
    cols: list[np.ndarray] = []
    for field in features:
        raw = np.array([number(rows[index].get(field)) for index in indices], dtype=float)
        missing = ~np.isfinite(raw)
        if prep is None:
            median = float(np.nanmedian(raw)) if np.isfinite(raw).any() else 0.0
            filled = np.where(missing, median, raw)
            mean = float(filled.mean())
            scale = float(filled.std()) or 1.0
            fitted[field] = (median, mean, scale)
        median, mean, scale = fitted[field]
        cols.append((np.where(missing, median, raw) - mean) / scale)
        if add_missing and field not in {name + "_missing" for name in SNAPSHOT_FEATURES}:
            cols.append(missing.astype(float))
    return (np.column_stack(cols) if cols else np.empty((len(indices), 0))), fitted


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def train_logistic(x: np.ndarray, y: np.ndarray, l2: float = 0.01) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(x1.shape[1])
    for _ in range(60):
        probability = sigmoid(x1 @ weights)
        variance = np.clip(probability * (1 - probability), 1e-6, None)
        gradient = x1.T @ (probability - y) / len(y)
        gradient[1:] += l2 * weights[1:]
        hessian = (x1.T * variance) @ x1 / len(y)
        hessian[1:, 1:] += l2 * np.eye(x.shape[1])
        step = np.linalg.solve(hessian + 1e-8 * np.eye(len(weights)), gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return weights


def predict_logistic(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return sigmoid(np.column_stack([np.ones(len(x)), x]) @ weights)


def temperature(raw: np.ndarray, y: np.ndarray) -> float:
    logits = np.log(np.clip(raw, 1e-7, 1 - 1e-7) / np.clip(1 - raw, 1e-7, 1))
    candidates = np.linspace(0.5, 5.0, 181)
    losses = []
    for value in candidates:
        p = sigmoid(logits / value)
        losses.append(-float(np.mean(y * np.log(np.clip(p, 1e-8, 1)) + (1-y) * np.log(np.clip(1-p, 1e-8, 1)))))
    return float(candidates[int(np.argmin(losses))])


def apply_temperature(raw: np.ndarray, value: float) -> np.ndarray:
    logits = np.log(np.clip(raw, 1e-7, 1 - 1e-7) / np.clip(1 - raw, 1e-7, 1))
    return sigmoid(logits / value)


def fit_predict(
    rows: list[dict[str, object]], train: list[int], calibration: list[int], test: list[int],
    features: list[str], l2: float = 0.01, add_missing: bool = False,
) -> tuple[np.ndarray, float, np.ndarray]:
    x_train, prep = design(rows, train, features, add_missing=add_missing)
    x_cal, _ = design(rows, calibration, features, prep, add_missing)
    x_test, _ = design(rows, test, features, prep, add_missing)
    weights = train_logistic(x_train, labels(rows, train), l2=l2)
    raw_cal = predict_logistic(x_cal, weights)
    value = temperature(raw_cal, labels(rows, calibration))
    return apply_temperature(predict_logistic(x_test, weights), value), value, weights


def train_boosted_stumps(
    x: np.ndarray, y: np.ndarray, rounds: int = 120, learning_rate: float = 0.08
) -> tuple[float, list[tuple[int, float, float, float]]]:
    """Fit a small nonlinear gradient-boosted stump model on logistic residuals."""
    prevalence = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    intercept = math.log(prevalence / (1 - prevalence))
    score = np.full(len(y), intercept)
    trees: list[tuple[int, float, float, float]] = []
    thresholds = [np.unique(np.quantile(x[:, j], np.linspace(0.1, 0.9, 9))) for j in range(x.shape[1])]
    for _ in range(rounds):
        residual = y - sigmoid(score)
        best: tuple[float, int, float, float, float] | None = None
        for feature, candidates in enumerate(thresholds):
            column = x[:, feature]
            for threshold_value in candidates:
                left = column <= threshold_value
                n_left = int(left.sum())
                if n_left < 12 or len(y) - n_left < 12:
                    continue
                left_value = float(residual[left].mean())
                right_value = float(residual[~left].mean())
                fitted = np.where(left, left_value, right_value)
                loss = float(np.mean((residual - fitted) ** 2))
                if best is None or loss < best[0]:
                    best = (loss, feature, float(threshold_value), left_value, right_value)
        if best is None:
            break
        _, feature, threshold_value, left_value, right_value = best
        trees.append((feature, threshold_value, learning_rate * left_value, learning_rate * right_value))
        score += np.where(x[:, feature] <= threshold_value, learning_rate * left_value, learning_rate * right_value)
    return intercept, trees


def predict_boosted_stumps(x: np.ndarray, model: tuple[float, list[tuple[int, float, float, float]]]) -> np.ndarray:
    intercept, trees = model
    score = np.full(len(x), intercept)
    for feature, threshold_value, left_value, right_value in trees:
        score += np.where(x[:, feature] <= threshold_value, left_value, right_value)
    return sigmoid(score)


def fit_predict_boosted(
    rows: list[dict[str, object]], train: list[int], calibration: list[int], test: list[int], features: list[str]
) -> tuple[np.ndarray, float]:
    x_train, prep = design(rows, train, features, add_missing=True)
    x_cal, _ = design(rows, calibration, features, prep, add_missing=True)
    x_test, _ = design(rows, test, features, prep, add_missing=True)
    model = train_boosted_stumps(x_train, labels(rows, train))
    raw_cal = predict_boosted_stumps(x_cal, model)
    value = temperature(raw_cal, labels(rows, calibration))
    return apply_temperature(predict_boosted_stumps(x_test, model), value), value


def split(rows: list[dict[str, object]], seed: int) -> tuple[list[int], list[int], list[int]]:
    development = [i for i, r in enumerate(rows) if r["temporal_role"] == "development_resolved_at_cutoff"]
    test = [i for i, r in enumerate(rows) if r["temporal_role"] == "future_resolved_test"]
    hosts = sorted({str(rows[i]["tic_id"]) for i in development})
    random.Random(seed).shuffle(hosts)
    cal_hosts = set(hosts[:max(1, round(0.25 * len(hosts)))])
    return (
        [i for i in development if rows[i]["tic_id"] not in cal_hosts],
        [i for i in development if rows[i]["tic_id"] in cal_hosts],
        test,
    )


def expanded_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    output = metrics(y, probability)
    prevalence = float(y.mean())
    order = np.argsort(probability)[::-1]
    positives = max(1, int(y.sum()))
    for k in KS:
        selected = order[:min(k, len(order))]
        p_at_k = float(y[selected].mean())
        output[f"recall_at_{k}"] = float(y[selected].sum() / positives)
        output[f"enrichment_at_{k}"] = p_at_k / prevalence
        output[f"queue_coverage_at_{k}"] = len(selected) / len(y)
    return output


def summary(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(float(row["value"]))
    output = []
    for group, values in sorted(grouped.items()):
        array = np.array(values)
        record: dict[str, object] = dict(zip(keys, group))
        record.update(n=len(array), mean=float(array.mean()), sd=float(array.std(ddof=1)) if len(array)>1 else 0.0,
                      q025=float(np.quantile(array, .025)), q975=float(np.quantile(array, .975)))
        output.append(record)
    return output


def subset_indices(rows: list[dict[str, object]], test: list[int], subset: str) -> list[int]:
    if subset == "all":
        return test
    if subset == "dv_available":
        return [i for i in test if rows[i]["event_dv_strict"] == 1]
    if subset == "xml_available":
        return [i for i in test if rows[i]["event_xml_strict"] == 1]
    if subset == "no_strict_product":
        return [i for i in test if rows[i]["event_dv_strict"] == 0]
    raise ValueError(subset)


def two_stage_bootstrap(
    rows: list[dict[str, object]], replicates: int, seed: int,
) -> list[dict[str, object]]:
    dev_hosts = sorted({str(r["tic_id"]) for r in rows if r["temporal_role"] == "development_resolved_at_cutoff"})
    test_hosts = sorted({str(r["tic_id"]) for r in rows if r["temporal_role"] == "future_resolved_test"})
    by_host: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_host[str(row["tic_id"])].append(i)
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    for replicate in range(replicates):
        shuffled = list(dev_hosts)
        random.Random(seed + replicate).shuffle(shuffled)
        n_cal = max(1, round(0.25 * len(shuffled)))
        cal_pool, train_pool = shuffled[:n_cal], shuffled[n_cal:]
        train_hosts = rng.choice(train_pool, len(train_pool), replace=True)
        cal_hosts = rng.choice(cal_pool, len(cal_pool), replace=True)
        sampled_test_hosts = rng.choice(test_hosts, len(test_hosts), replace=True)
        train = [i for host in train_hosts for i in by_host[str(host)] if rows[i]["temporal_role"] == "development_resolved_at_cutoff"]
        calibration = [i for host in cal_hosts for i in by_host[str(host)] if rows[i]["temporal_role"] == "development_resolved_at_cutoff"]
        test = [i for host in sampled_test_hosts for i in by_host[str(host)] if rows[i]["temporal_role"] == "future_resolved_test"]
        if len(set(labels(rows, train))) < 2 or len(set(labels(rows, calibration))) < 2 or len(set(labels(rows, test))) < 2:
            continue
        predictions = {}
        for model in ("catalog", "all_strict"):
            predictions[model] = fit_predict(rows, train, calibration, test, FEATURE_SETS[model])[0]
        y = labels(rows, test)
        base, full = expanded_metrics(y, predictions["catalog"]), expanded_metrics(y, predictions["all_strict"])
        for metric in ("brier", "ece", "average_precision", "roc_auc", "precision_at_10", "precision_at_50", "precision_at_100"):
            value = base[metric] - full[metric] if metric in LOWER_IS_BETTER else full[metric] - base[metric]
            output.append({"replicate": replicate, "metric": metric, "value": value})
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--dv", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=300)
    args = parser.parse_args()

    rows = annotate_event_association(merge(read_csv(pathlib.Path(args.cohort)), read_csv(pathlib.Path(args.dv)), read_csv(pathlib.Path(args.xml))))
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    provenance_fields = [
        "tic_id", "toi", "temporal_role", "current_disposition", "dv_selected_tce", "dv_selected_sector",
        "historic_period_days", "historic_epoch_btjd", "dv_data_tperiod", "dv_data_tepoch",
        "xml_planetcandidate_orbitalperiodindays", "xml_planetcandidate_epochtjd",
        "xml_planetcandidate_planetnumber", "xml_planetresults_toiid", "xml_dvtargetresults_toiid",
        "dv_product_filename", "dv_product_last_modified",
        "event_dv_period_rel_error", "event_dv_epoch_phase_error_days", "event_xml_period_rel_error",
        "event_xml_epoch_phase_error_days", "event_epoch_tolerance_days", "event_dv_strict",
        "event_xml_strict", "event_ambiguous", "event_ambiguity_reason",
    ]
    write_rows(outdir / "event_level_provenance.csv", [{field: row.get(field, "") for field in provenance_fields} for row in rows])

    repeated: list[dict[str, object]] = []
    fixed: dict[str, np.ndarray] = {}
    fixed_test: list[int] = []
    coefficient_rows: list[dict[str, object]] = []
    for seed in range(11, 11 + args.n_seeds):
        train, calibration, test = split(rows, seed)
        y_test = labels(rows, test)
        for model, features in FEATURE_SETS.items():
            probability, temp, weights = fit_predict(rows, train, calibration, test, features)
            for metric, value in expanded_metrics(y_test, probability).items():
                repeated.append({"seed": seed, "model": model, "metric": metric, "value": value, "temperature": temp})
            if seed == 11:
                fixed[model] = probability
                fixed_test = test
        probability, temp, weights = fit_predict(rows, train, calibration, test, FEATURE_SETS["all_strict"], add_missing=True)
        for metric, value in expanded_metrics(y_test, probability).items():
            repeated.append({"seed": seed, "model": "all_strict_field_missingness", "metric": metric, "value": value, "temperature": temp})
        for l2 in (0.001, 0.01, 0.1):
            probability, temp, weights = fit_predict(rows, train, calibration, test, FEATURE_SETS["all_strict"], l2=l2)
            for metric, value in expanded_metrics(y_test, probability).items():
                repeated.append({"seed": seed, "model": f"all_strict_l2_{l2:g}", "metric": metric, "value": value, "temperature": temp})
            coefficient_rows.append({"seed": seed, "l2": l2, "temperature": temp, "coefficient_l2_norm": float(np.linalg.norm(weights[1:])), "max_abs_coefficient": float(np.max(np.abs(weights[1:])))})
        for model in ("catalog", "all_strict"):
            probability, temp = fit_predict_boosted(rows, train, calibration, test, FEATURE_SETS[model])
            for metric, value in expanded_metrics(y_test, probability).items():
                repeated.append({"seed": seed, "model": model + "_boosted_stumps", "metric": metric, "value": value, "temperature": temp})

    write_rows(outdir / "repeated_sensitivity_metrics.csv", repeated)
    repeated_summary = summary(repeated, ("model", "metric"))
    write_rows(outdir / "repeated_sensitivity_summary.csv", repeated_summary)
    write_rows(outdir / "coefficient_regularization_sensitivity.csv", coefficient_rows)

    # Same-row and complete-case contrasts use fixed seed-11 predictions.  The
    # complete-case refit additionally removes missing-product interactions from fitting.
    subset_rows: list[dict[str, object]] = []
    local_by_global = {index: local for local, index in enumerate(fixed_test)}
    for subset in ("all", "dv_available", "xml_available", "no_strict_product"):
        indices = subset_indices(rows, fixed_test, subset)
        local = np.array([local_by_global[i] for i in indices], dtype=int)
        y = labels(rows, indices)
        if len(set(y)) < 2:
            continue
        for model in ("catalog", "catalog_availability", "all_strict"):
            for metric, value in expanded_metrics(y, fixed[model][local]).items():
                subset_rows.append({"analysis": "same_fitted_models", "subset": subset, "model": model, "metric": metric, "value": value, "n": len(indices), "prevalence": float(y.mean())})

    train, calibration, test = split(rows, 11)
    for availability, condition in (("dv_available", "event_dv_strict"), ("xml_available", "event_xml_strict")):
        tr = [i for i in train if rows[i][condition] == 1]
        ca = [i for i in calibration if rows[i][condition] == 1]
        te = [i for i in test if rows[i][condition] == 1]
        for model in ("catalog", "all_strict"):
            probability = fit_predict(rows, tr, ca, te, FEATURE_SETS[model])[0]
            for metric, value in expanded_metrics(labels(rows, te), probability).items():
                subset_rows.append({"analysis": "complete_case_refit", "subset": availability, "model": model, "metric": metric, "value": value, "n": len(te), "prevalence": float(labels(rows, te).mean())})
    write_rows(outdir / "missingness_stratified_metrics.csv", subset_rows)

    # Negative-category sensitivity and an explicitly evaluated score-margin abstention rule.
    category_rows: list[dict[str, object]] = []
    y_all = labels(rows, fixed_test)
    for model in ("catalog", "all_strict"):
        probability = fixed[model]
        for category in ("EB", "V", "IS", "O"):
            local = np.array([j for j, i in enumerate(fixed_test) if y_all[j] == 1 or rows[i]["current_disposition"] == category], dtype=int)
            category_rows.append({"model": model, "negative_category": category, "n": len(local), "n_negative": int((y_all[local] == 0).sum()), "roc_auc": roc_auc(y_all[local], probability[local]), "average_precision": average_precision(y_all[local], probability[local])})
    write_rows(outdir / "label_category_sensitivity.csv", category_rows)

    abstention_rows: list[dict[str, object]] = []
    for model in ("catalog", "all_strict"):
        probability = fixed[model]
        confidence_order = np.argsort(np.abs(probability - 0.5))[::-1]
        for coverage in (0.25, 0.50, 0.75, 1.0):
            selected = confidence_order[:max(1, round(coverage * len(probability)))]
            predicted = (probability[selected] >= 0.5).astype(float)
            abstention_rows.append({"model": model, "coverage": coverage, "n_retained": len(selected), "error_rate": float((predicted != y_all[selected]).mean()), "ece": ece(y_all[selected], probability[selected]), "positive_rate": float(y_all[selected].mean())})
    write_rows(outdir / "abstention_margin_tradeoff.csv", abstention_rows)

    nested = two_stage_bootstrap(rows, args.bootstrap_replicates, 20210719)
    write_rows(outdir / "two_stage_host_bootstrap.csv", nested)
    nested_summary = summary(nested, ("metric",))
    write_rows(outdir / "two_stage_host_bootstrap_summary.csv", nested_summary)

    event_counts = {
        "rows": len(rows),
        "period_only_dv_matches": sum(str(r.get("dv_candidate_match")) in {"1", "1.0"} for r in rows),
        "strict_dv_matches": sum(r["event_dv_strict"] == 1 for r in rows),
        "xml_records_found": sum(str(r.get("xml_found")) in {"1", "1.0"} for r in rows),
        "strict_xml_matches": sum(r["event_xml_strict"] == 1 for r in rows),
        "ambiguous_rows": sum(r["event_ambiguous"] == 1 for r in rows),
        "ambiguity_reasons": dict(Counter(str(r["event_ambiguity_reason"]) for r in rows if r["event_ambiguity_reason"])),
        "duplicate_strict_xml_products": sum(count > 1 for count in Counter(str(r.get("dv_product_filename")) for r in rows if r["event_xml_strict"] == 1).values()),
    }
    report = {"endpoint": "later TOI disposition (not independent planet truth or RV yield)", "event_association": event_counts, "repeated_summary": repeated_summary, "nested_bootstrap_summary": nested_summary}
    (outdir / "acceptance_analysis_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Acceptance analyses", "", "The endpoint is later TOI disposition, not independent planet truth or realized RV yield.", "", "## Event association", ""]
    lines.extend(f"- {key.replace('_', ' ').capitalize()}: {value}" for key, value in event_counts.items() if key != "ambiguity_reasons")
    lines.extend(["", "## Two-stage host bootstrap: all strict evidence minus catalog", "", "| Metric | Mean improvement [95% interval] |", "|---|---:|"])
    for row in nested_summary:
        lines.append(f"| {row['metric']} | {row['mean']:.4f} [{row['q025']:.4f}, {row['q975']:.4f}] |")
    lines.extend(["", "Full tables include repeated splits, complete-case and availability strata, label-category sensitivity, regularization sensitivity, and score-margin abstention."])
    (outdir / "acceptance_analysis_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(event_counts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
