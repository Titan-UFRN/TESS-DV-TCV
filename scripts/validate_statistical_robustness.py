#!/usr/bin/env python3
"""Quantify split, sampling, policy, and unresolved-ranking uncertainty."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import pathlib
from collections import Counter, defaultdict

import numpy as np

from revise_candidate_scoring import needs_manual_review, risk_notes
from tune_vetting_penalty import GRID, GROUPS, hand_penalty, penalty_from_groups, rv_utility
from validate_ablation_calibration import (
    FEATURE_SETS,
    apply_temperature,
    available_features,
    brier,
    calibrate_temperature,
    ece,
    fit_transform,
    precision_at_k,
    predict,
    read_csv,
    split_indices,
    train_l2,
    transform,
    y_for,
)


KS = (10, 25, 50, 100, 250)
POLICY_KS = (50, 100, 250)
SCALAR_GRID = np.linspace(0.20, 1.00, 17)


def write_rows(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nll(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p, 1e-8, 1 - 1e-8)
    return -float(np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(score, kind="mergesort")[::-1]
    ranked_y = y[order]
    precision = np.cumsum(ranked_y) / np.arange(1, len(ranked_y) + 1)
    return float(precision[ranked_y == 1].sum() / n_pos)


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {
        "brier": brier(y, p),
        "ece": ece(y, p),
        "nll": nll(y, p),
        "roc_auc": roc_auc(y, p),
        "average_precision": average_precision(y, p),
    }
    for k in KS:
        value = precision_at_k(y, p, k)
        out[f"precision_at_{k}"] = float(value) if value is not None else math.nan
    return out


def fit_predictions(
    rows: list[dict[str, str]],
    train_idx: list[int],
    cal_idx: list[int],
    test_idx: list[int],
    feature_names: list[str],
    predict_idx: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float, list[str]]:
    features = available_features(rows, feature_names)
    x_train, stats = fit_transform(rows, train_idx, features)
    weights = train_l2(x_train, y_for(rows, train_idx))
    p_cal_raw = predict(transform(rows, cal_idx, features, stats), weights)
    temperature = calibrate_temperature(p_cal_raw, y_for(rows, cal_idx))
    p_cal = apply_temperature(p_cal_raw, temperature)
    p_test = apply_temperature(predict(transform(rows, test_idx, features, stats), weights), temperature)
    p_extra = None
    if predict_idx is not None:
        p_extra = apply_temperature(predict(transform(rows, predict_idx, features, stats), weights), temperature)
    return p_cal, p_test, p_extra, temperature, features


def summarize_rows(rows: list[dict[str, object]], group_fields: tuple[str, ...], value_fields: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    out: list[dict[str, object]] = []
    for group, items in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        result: dict[str, object] = dict(zip(group_fields, group))
        result["n"] = len(items)
        for field in value_fields:
            values = np.array([float(item[field]) for item in items], dtype=np.float64)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            result[f"{field}_mean"] = float(values.mean())
            result[f"{field}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            result[f"{field}_median"] = float(np.median(values))
            result[f"{field}_q025"] = float(np.quantile(values, 0.025))
            result[f"{field}_q975"] = float(np.quantile(values, 0.975))
        out.append(result)
    return out


def group_patterns(notes: list[list[str]]) -> np.ndarray:
    group_items = list(GROUPS.items())
    matrix = np.zeros((len(notes), len(group_items)), dtype=np.float64)
    for i, row_notes in enumerate(notes):
        note_set = set(row_notes)
        for j, (_, group_notes) in enumerate(group_items):
            matrix[i, j] = float(bool(note_set & group_notes))
    return matrix


def tune_weights_vectorized(
    y: np.ndarray,
    base_score: np.ndarray,
    risks: np.ndarray,
    notes: list[list[str]],
    batch_size: int,
) -> tuple[dict[str, float], float]:
    group_names = list(GROUPS)
    patterns = group_patterns(notes)
    combinations = np.array(list(itertools.product(GRID, repeat=len(group_names))), dtype=np.float64)
    best_objective = -math.inf
    best_index = 0
    k50 = min(50, len(y))
    k100 = min(100, len(y))
    for start in range(0, len(combinations), batch_size):
        weights = combinations[start : start + batch_size]
        penalties = np.exp(np.log(weights) @ patterns.T)
        penalties = np.clip(penalties, 0.02, 1.0)
        scores = penalties * base_score[None, :]
        top50 = np.argpartition(scores, -k50, axis=1)[:, -k50:]
        top100 = np.argpartition(scores, -k100, axis=1)[:, -k100:]
        p50 = y[top50].mean(axis=1)
        p100 = y[top100].mean(axis=1)
        risk100 = risks[top100].mean(axis=1)
        objectives = p100 + 0.25 * p50 - 0.20 * risk100
        local_index = int(np.argmax(objectives))
        local_value = float(objectives[local_index])
        if local_value > best_objective:
            best_objective = local_value
            best_index = start + local_index
    return dict(zip(group_names, combinations[best_index].tolist())), best_objective


def policy_penalties(notes: list[list[str]], weights: dict[str, float] | None) -> np.ndarray:
    if weights is None:
        return np.array([hand_penalty(item) for item in notes], dtype=np.float64)
    return np.array([penalty_from_groups(item, weights) for item in notes], dtype=np.float64)


def scalar_penalties(notes: list[list[str]], weight: float) -> np.ndarray:
    counts = group_patterns(notes).sum(axis=1)
    return np.clip(weight**counts, 0.02, 1.0)


def policy_objective(y: np.ndarray, scores: np.ndarray, risks: np.ndarray) -> float:
    order50 = np.argsort(scores, kind="mergesort")[::-1][: min(50, len(scores))]
    order100 = np.argsort(scores, kind="mergesort")[::-1][: min(100, len(scores))]
    return float(y[order100].mean() + 0.25 * y[order50].mean() - 0.20 * risks[order100].mean())


def tune_scalar_weight(
    y: np.ndarray,
    base_score: np.ndarray,
    risks: np.ndarray,
    notes: list[list[str]],
) -> tuple[float, float]:
    best_weight = 1.0
    best_objective = -math.inf
    for weight in SCALAR_GRID:
        value = policy_objective(y, base_score * scalar_penalties(notes, float(weight)), risks)
        if value > best_objective:
            best_weight = float(weight)
            best_objective = value
    return best_weight, best_objective


def policy_metrics(y: np.ndarray, score: np.ndarray, risks: np.ndarray) -> dict[str, object]:
    out: dict[str, object] = {}
    for k in POLICY_KS:
        order = np.argsort(score, kind="mergesort")[::-1][: min(k, len(score))]
        out[f"precision_at_{k}"] = float(y[order].mean())
        out[f"manual_rate_at_{k}"] = float(risks[order].mean())
        out[f"manual_count_at_{k}"] = int(risks[order].sum())
    return out


def cluster_bootstrap(
    rows: list[dict[str, str]],
    test_idx: list[int],
    predictions: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    host_to_local: dict[str, list[int]] = defaultdict(list)
    for local_i, row_i in enumerate(test_idx):
        host_to_local[str(rows[row_i].get("tic_id") or row_i)].append(local_i)
    hosts = sorted(host_to_local)
    clusters = [np.array(host_to_local[host], dtype=int) for host in hosts]
    y = y_for(rows, test_idx)
    rng = np.random.default_rng(seed)
    metric_samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    paired_samples: dict[str, list[float]] = defaultdict(list)

    for _ in range(replicates):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        local_idx = np.concatenate([clusters[i] for i in sampled])
        y_boot = y[local_idx]
        values: dict[str, dict[str, float]] = {}
        for model, p in predictions.items():
            values[model] = metric_values(y_boot, p[local_idx])
            for metric, value in values[model].items():
                metric_samples[(model, metric)].append(value)
        baseline = values["archive_tic"]
        full = values["all_features"]
        for metric in baseline:
            improvement = baseline[metric] - full[metric] if metric in {"brier", "ece", "nll"} else full[metric] - baseline[metric]
            paired_samples[metric].append(improvement)

    intervals: list[dict[str, object]] = []
    for (model, metric), values in sorted(metric_samples.items()):
        arr = np.array(values, dtype=np.float64)
        intervals.append({
            "model": model,
            "metric": metric,
            "replicates": replicates,
            "mean": float(arr.mean()),
            "q025": float(np.quantile(arr, 0.025)),
            "q975": float(np.quantile(arr, 0.975)),
        })
    paired: list[dict[str, object]] = []
    for metric, values in sorted(paired_samples.items()):
        arr = np.array(values, dtype=np.float64)
        paired.append({
            "comparison": "all_features_vs_archive_tic",
            "metric": metric,
            "positive_favors_all_features": 1,
            "replicates": replicates,
            "mean_improvement": float(arr.mean()),
            "q025": float(np.quantile(arr, 0.025)),
            "q975": float(np.quantile(arr, 0.975)),
            "positive_fraction": float((arr > 0).mean()),
        })
    return intervals, paired


def ranking_stability_summary(top_sets: dict[str, list[set[int]]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for policy, sets in top_sets.items():
        values = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                values.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
        arr = np.array(values, dtype=np.float64)
        counts = Counter(item for selected in sets for item in selected)
        out.append({
            "policy": policy,
            "top_k": 50,
            "pair_count": len(values),
            "jaccard_mean": float(arr.mean()) if len(arr) else 1.0,
            "jaccard_median": float(np.median(arr)) if len(arr) else 1.0,
            "jaccard_q025": float(np.quantile(arr, 0.025)) if len(arr) else 1.0,
            "jaccard_q975": float(np.quantile(arr, 0.975)) if len(arr) else 1.0,
            "top50_union_count": len(counts),
            "top50_always_count": sum(count == len(sets) for count in counts.values()),
            "top50_frequency_ge_080_count": sum(count / len(sets) >= 0.80 for count in counts.values()),
            "top50_frequency_ge_050_count": sum(count / len(sets) >= 0.50 for count in counts.values()),
        })
    return out


def paired_policy_summary(policy_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_seed: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in policy_rows:
        by_seed[int(row["seed"])][str(row["policy"])] = row
    out: list[dict[str, object]] = []
    for policy in ("hand_penalty", "scalar_tuned", "tuned"):
        for metric in ("precision_at_100", "precision_at_250", "manual_count_at_100", "manual_count_at_250"):
            values = []
            for seed in sorted(by_seed):
                original = float(by_seed[seed]["original"][metric])
                current = float(by_seed[seed][policy][metric])
                values.append(original - current if metric.startswith("manual_count") else current - original)
            arr = np.array(values, dtype=np.float64)
            out.append({
                "comparison": f"{policy}_vs_original",
                "metric": metric,
                "positive_favors_policy": 1,
                "n_splits": len(arr),
                "mean_improvement": float(arr.mean()),
                "q025": float(np.quantile(arr, 0.025)),
                "q975": float(np.quantile(arr, 0.975)),
                "positive_fraction": float((arr > 0).mean()),
                "nonnegative_fraction": float((arr >= 0).mean()),
            })
    return out


def markdown_report(
    path: pathlib.Path,
    split_summary: list[dict[str, object]],
    paired_bootstrap: list[dict[str, object]],
    policy_summary: list[dict[str, object]],
    paired_policy: list[dict[str, object]],
    ranking_summary: list[dict[str, object]],
    seeds: list[int],
    bootstrap_replicates: int,
) -> None:
    by_model = {str(row["model"]): row for row in split_summary}
    by_policy = {str(row["policy"]): row for row in policy_summary}
    paired = {str(row["metric"]): row for row in paired_bootstrap}
    lines = [
        "# Statistical Robustness Report",
        "",
        f"Repeated TIC-grouped splits: {len(seeds)} (seeds {min(seeds)}--{max(seeds)}).",
        f"Fixed-test TIC-cluster bootstrap replicates: {bootstrap_replicates}.",
        "",
        "## Repeated-split model summary",
        "",
        "| Model | Brier mean [95% split interval] | ECE mean [95% split interval] | AP mean [95% split interval] | P@100 mean [95% split interval] |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in FEATURE_SETS:
        row = by_model[model]
        lines.append(
            f"| `{model}` | {row['brier_mean']:.4f} [{row['brier_q025']:.4f}, {row['brier_q975']:.4f}] | "
            f"{row['ece_mean']:.4f} [{row['ece_q025']:.4f}, {row['ece_q975']:.4f}] | "
            f"{row['average_precision_mean']:.4f} [{row['average_precision_q025']:.4f}, {row['average_precision_q975']:.4f}] | "
            f"{row['precision_at_100_mean']:.3f} [{row['precision_at_100_q025']:.3f}, {row['precision_at_100_q975']:.3f}] |"
        )
    lines.extend([
        "",
        "## Paired cluster-bootstrap improvements",
        "",
        "Positive values favor the all-feature model over archive/TIC.",
        "",
        "| Metric | Mean improvement | 95% cluster-bootstrap interval | Positive fraction |",
        "|---|---:|---:|---:|",
    ])
    for metric in ("brier", "ece", "nll", "average_precision", "roc_auc", "precision_at_100"):
        row = paired[metric]
        lines.append(
            f"| `{metric}` | {row['mean_improvement']:.4f} | [{row['q025']:.4f}, {row['q975']:.4f}] | {row['positive_fraction']:.3f} |"
        )
    lines.extend([
        "",
        "## Nested repeated policy summary",
        "",
        "Each tuned policy is selected on that split's calibration hosts and evaluated on its held-out test hosts.",
        "",
        "| Policy | P@100 mean [95% split interval] | Manual@100 mean [95% split interval] |",
        "|---|---:|---:|",
    ])
    for policy in ("original", "hand_penalty", "scalar_tuned", "tuned"):
        row = by_policy[policy]
        lines.append(
            f"| `{policy}` | {row['precision_at_100_mean']:.3f} [{row['precision_at_100_q025']:.3f}, {row['precision_at_100_q975']:.3f}] | "
            f"{row['manual_count_at_100_mean']:.1f} [{row['manual_count_at_100_q025']:.1f}, {row['manual_count_at_100_q975']:.1f}] |"
        )
    paired_policy_map = {(str(row["comparison"]), str(row["metric"])): row for row in paired_policy}
    lines.extend([
        "",
        "## Paired policy differences across splits",
        "",
        "Positive values favor the named policy; split intervals describe partition sensitivity and are not independent-sample confidence intervals.",
        "",
        "| Comparison | P@100 mean difference [95% split interval] | Manual@100 mean reduction [95% split interval] |",
        "|---|---:|---:|",
    ])
    for policy in ("hand_penalty", "scalar_tuned", "tuned"):
        precision = paired_policy_map[(f"{policy}_vs_original", "precision_at_100")]
        manual = paired_policy_map[(f"{policy}_vs_original", "manual_count_at_100")]
        lines.append(
            f"| `{policy}` vs original | {precision['mean_improvement']:.3f} [{precision['q025']:.3f}, {precision['q975']:.3f}] | "
            f"{manual['mean_improvement']:.1f} [{manual['q025']:.1f}, {manual['q975']:.1f}] |"
        )
    lines.extend([
        "",
        "## Unresolved top-50 stability",
        "",
        "| Policy | Mean pairwise Jaccard | 95% pairwise interval | Candidates present in at least 80% of top-50 queues |",
        "|---|---:|---:|---:|",
    ])
    for row in ranking_summary:
        lines.append(
            f"| `{row['policy']}` | {row['jaccard_mean']:.3f} | [{row['jaccard_q025']:.3f}, {row['jaccard_q975']:.3f}] | {row['top50_frequency_ge_080_count']} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", default="data/processed/full_dv_xml_merged.csv")
    parser.add_argument("--outdir", default="reports/robustness")
    parser.add_argument("--seed-start", type=int, default=11)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--tuning-batch-size", type=int, default=2048)
    args = parser.parse_args()

    rows = read_csv(pathlib.Path(args.merged))
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    candidate_idx = [
        i for i, row in enumerate(rows)
        if row.get("strict_planet_class") in {"planet_candidate", "ambiguous_planet_candidate"}
    ]
    candidate_notes = [risk_notes(rows[i]) for i in candidate_idx]
    candidate_utility = np.array([rv_utility(rows[i]) for i in candidate_idx], dtype=np.float64)
    candidate_hand = policy_penalties(candidate_notes, None)

    split_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    fixed_predictions: dict[str, np.ndarray] = {}
    fixed_test_idx: list[int] = []
    candidate_probabilities: list[np.ndarray] = []
    candidate_scores: dict[str, list[np.ndarray]] = {name: [] for name in ("original", "hand_penalty", "scalar_tuned", "tuned")}
    candidate_ranks: dict[str, list[np.ndarray]] = {name: [] for name in candidate_scores}
    top_sets: dict[str, list[set[int]]] = {name: [] for name in candidate_scores}

    for seed_number, seed in enumerate(seeds, start=1):
        train_idx, cal_idx, test_idx = split_indices(rows, seed=seed)
        y_test = y_for(rows, test_idx)
        all_outputs: tuple[np.ndarray, np.ndarray, np.ndarray | None, float, list[str]] | None = None
        for model, feature_names in FEATURE_SETS.items():
            output = fit_predictions(
                rows,
                train_idx,
                cal_idx,
                test_idx,
                feature_names,
                candidate_idx if model == "all_features" else None,
            )
            p_cal, p_test, p_candidate, temperature, features = output
            metrics = metric_values(y_test, p_test)
            split_rows.append({
                "seed": seed,
                "model": model,
                "n_features": len(features),
                "n_train": len(train_idx),
                "n_calibration": len(cal_idx),
                "n_test": len(test_idx),
                "temperature": temperature,
                **metrics,
            })
            if seed == seeds[0]:
                fixed_predictions[model] = p_test
                fixed_test_idx = test_idx
            if model == "all_features":
                all_outputs = output

        if all_outputs is None or all_outputs[2] is None:
            raise RuntimeError("all-feature candidate predictions were not produced")
        p_cal, p_test, p_candidate, _, _ = all_outputs
        y_cal = y_for(rows, cal_idx)
        y_test = y_for(rows, test_idx)
        cal_notes = [risk_notes(rows[i]) for i in cal_idx]
        test_notes = [risk_notes(rows[i]) for i in test_idx]
        cal_risks = np.array([float(needs_manual_review(notes)) for notes in cal_notes])
        test_risks = np.array([float(needs_manual_review(notes)) for notes in test_notes])
        cal_utility = np.array([rv_utility(rows[i]) for i in cal_idx], dtype=np.float64)
        test_utility = np.array([rv_utility(rows[i]) for i in test_idx], dtype=np.float64)
        tuned_weights, calibration_objective = tune_weights_vectorized(
            y_cal,
            p_cal * cal_utility,
            cal_risks,
            cal_notes,
            batch_size=args.tuning_batch_size,
        )
        scalar_weight, scalar_objective = tune_scalar_weight(
            y_cal,
            p_cal * cal_utility,
            cal_risks,
            cal_notes,
        )
        weight_rows.append({
            "seed": seed,
            "calibration_objective": calibration_objective,
            "scalar_weight": scalar_weight,
            "scalar_calibration_objective": scalar_objective,
            **tuned_weights,
        })

        test_base = p_test * test_utility
        policy_test_scores = {
            "original": test_base,
            "hand_penalty": test_base * policy_penalties(test_notes, None),
            "scalar_tuned": test_base * scalar_penalties(test_notes, scalar_weight),
            "tuned": test_base * policy_penalties(test_notes, tuned_weights),
        }
        for policy, scores in policy_test_scores.items():
            policy_rows.append({"seed": seed, "policy": policy, **policy_metrics(y_test, scores, test_risks)})

        candidate_probabilities.append(p_candidate)
        policy_candidate_scores = {
            "original": p_candidate * candidate_utility,
            "hand_penalty": p_candidate * candidate_utility * candidate_hand,
            "scalar_tuned": p_candidate * candidate_utility * scalar_penalties(candidate_notes, scalar_weight),
            "tuned": p_candidate * candidate_utility * policy_penalties(candidate_notes, tuned_weights),
        }
        for policy, scores in policy_candidate_scores.items():
            order = np.argsort(scores, kind="mergesort")[::-1]
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(1, len(order) + 1)
            candidate_scores[policy].append(scores)
            candidate_ranks[policy].append(ranks)
            top_sets[policy].append(set(order[:50].tolist()))
        print(f"completed seed {seed} ({seed_number}/{len(seeds)})", flush=True)

    metric_fields = list(metric_values(np.array([0.0, 1.0]), np.array([0.25, 0.75])))
    split_summary = summarize_rows(split_rows, ("model",), ["temperature", *metric_fields])
    policy_value_fields = [
        f"{name}_at_{k}"
        for k in POLICY_KS
        for name in ("precision", "manual_rate", "manual_count")
    ]
    policy_summary = summarize_rows(policy_rows, ("policy",), policy_value_fields)
    paired_policy = paired_policy_summary(policy_rows)

    bootstrap_intervals, paired_bootstrap = cluster_bootstrap(
        rows,
        fixed_test_idx,
        fixed_predictions,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )

    probability_matrix = np.vstack(candidate_probabilities)
    score_matrices = {policy: np.vstack(values) for policy, values in candidate_scores.items()}
    rank_matrices = {policy: np.vstack(values) for policy, values in candidate_ranks.items()}
    stability_rows: list[dict[str, object]] = []
    for local_i, row_i in enumerate(candidate_idx):
        source = rows[row_i]
        out: dict[str, object] = {
            "row_index": row_i,
            "tic_id": source.get("tic_id", ""),
            "toi": source.get("toi", ""),
            "strict_planet_class": source.get("strict_planet_class", ""),
            "expected_k_ms": source.get("expected_k_ms", ""),
            "planet_probability_mean": float(probability_matrix[:, local_i].mean()),
            "planet_probability_sd": float(probability_matrix[:, local_i].std(ddof=1)),
            "planet_probability_q025": float(np.quantile(probability_matrix[:, local_i], 0.025)),
            "planet_probability_q975": float(np.quantile(probability_matrix[:, local_i], 0.975)),
        }
        for policy in candidate_scores:
            ranks = rank_matrices[policy][:, local_i]
            scores = score_matrices[policy][:, local_i]
            out[f"{policy}_score_mean"] = float(scores.mean())
            out[f"{policy}_rank_median"] = float(np.median(ranks))
            out[f"{policy}_rank_q025"] = float(np.quantile(ranks, 0.025))
            out[f"{policy}_rank_q975"] = float(np.quantile(ranks, 0.975))
            out[f"{policy}_top50_count"] = int((ranks <= 50).sum())
            out[f"{policy}_top50_frequency"] = float((ranks <= 50).mean())
        stability_rows.append(out)
    stability_rows.sort(key=lambda row: (-float(row["tuned_top50_frequency"]), float(row["tuned_rank_median"])))
    ranking_summary = ranking_stability_summary(top_sets)

    weight_frequency: list[dict[str, object]] = []
    for group in GROUPS:
        counts = Counter(float(row[group]) for row in weight_rows)
        for value in GRID:
            weight_frequency.append({
                "group": group,
                "weight": value,
                "count": counts.get(float(value), 0),
                "frequency": counts.get(float(value), 0) / len(weight_rows),
            })

    write_rows(outdir / "repeated_split_metrics.csv", split_rows)
    write_rows(outdir / "repeated_split_summary.csv", split_summary)
    write_rows(outdir / "cluster_bootstrap_intervals.csv", bootstrap_intervals)
    write_rows(outdir / "paired_cluster_bootstrap_improvements.csv", paired_bootstrap)
    write_rows(outdir / "policy_repeated_metrics.csv", policy_rows)
    write_rows(outdir / "policy_repeated_summary.csv", policy_summary)
    write_rows(outdir / "paired_policy_improvements.csv", paired_policy)
    write_rows(outdir / "tuned_weights_by_seed.csv", weight_rows)
    write_rows(outdir / "tuned_weight_frequency.csv", weight_frequency)
    write_rows(outdir / "candidate_ranking_stability.csv", stability_rows)
    write_rows(outdir / "ranking_stability_summary.csv", ranking_summary)

    summary = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "split_summary": split_summary,
        "cluster_bootstrap_intervals": bootstrap_intervals,
        "paired_cluster_bootstrap_improvements": paired_bootstrap,
        "policy_summary": policy_summary,
        "paired_policy_improvements": paired_policy,
        "ranking_stability_summary": ranking_summary,
        "tuned_weight_frequency": weight_frequency,
    }
    (outdir / "statistical_robustness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_report(
        outdir / "statistical_robustness_report.md",
        split_summary,
        paired_bootstrap,
        policy_summary,
        paired_policy,
        ranking_summary,
        seeds,
        args.bootstrap_replicates,
    )
    print(json.dumps({
        "n_seeds": len(seeds),
        "bootstrap_replicates": args.bootstrap_replicates,
        "outdir": str(outdir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
