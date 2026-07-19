#!/usr/bin/env python3
"""Audit train/calibration/test leakage for the strict-label split."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
from collections import Counter, defaultdict


STRICT_CLASSES = {"known_or_confirmed_planet", "false_positive_or_false_alarm"}


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def strict_indices(rows: list[dict[str, str]]) -> list[int]:
    return [i for i, row in enumerate(rows) if row.get("strict_planet_class") in STRICT_CLASSES]


def host_id(row: dict[str, str], fallback: int) -> str:
    tic = str(row.get("tic_id", "")).strip()
    return tic if tic else f"row:{fallback}"


def split_indices(rows: list[dict[str, str]], seed: int = 11) -> tuple[list[int], list[int], list[int]]:
    idx = strict_indices(rows)
    hosts = sorted({host_id(rows[i], i) for i in idx})
    random.Random(seed).shuffle(hosts)
    n_test = max(1, int(len(hosts) * 0.25))
    n_cal = max(1, int(len(hosts) * 0.20))
    test_hosts = set(hosts[:n_test])
    cal_hosts = set(hosts[n_test : n_test + n_cal])
    train, cal, test = [], [], []
    for i in idx:
        host = host_id(rows[i], i)
        if host in test_hosts:
            test.append(i)
        elif host in cal_hosts:
            cal.append(i)
        else:
            train.append(i)
    return train, cal, test


def split_summary(rows: list[dict[str, str]], idx: list[int]) -> dict[str, object]:
    classes = Counter(rows[i].get("strict_planet_class", "") for i in idx)
    hosts = {host_id(rows[i], i) for i in idx}
    missing_tic = sum(1 for i in idx if not str(rows[i].get("tic_id", "")).strip())
    duplicate_host_rows = len(idx) - len(hosts)
    return {
        "rows": len(idx),
        "hosts": len(hosts),
        "missing_tic_rows": missing_tic,
        "duplicate_host_rows_within_split": duplicate_host_rows,
        "class_counts": dict(sorted(classes.items())),
    }


def overlap(a: set[str], b: set[str]) -> list[str]:
    return sorted(a & b)


def write_markdown(path: pathlib.Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlaps = report["host_overlaps"]  # type: ignore[index]
    split = report["splits"]  # type: ignore[index]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Split Leakage Audit\n\n")
        f.write(f"Run date: {report['run_date']}\n\n")
        f.write(f"Input: `{report['input']}`\n\n")
        f.write(f"Seed: `{report['seed']}`\n\n")
        f.write(f"Leakage status: **{report['leakage_status']}**\n\n")
        f.write("## Split Counts\n\n")
        f.write("| Split | Rows | Hosts | Missing TIC rows | Duplicate host rows | Known/confirmed | False positive/false alarm |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for name in ("train", "calibration", "test"):
            s = split[name]  # type: ignore[index]
            classes = s["class_counts"]  # type: ignore[index]
            f.write(
                f"| {name} | {s['rows']} | {s['hosts']} | {s['missing_tic_rows']} | "
                f"{s['duplicate_host_rows_within_split']} | "
                f"{classes.get('known_or_confirmed_planet', 0)} | "
                f"{classes.get('false_positive_or_false_alarm', 0)} |\n"
            )
        f.write("\n## Host Overlaps\n\n")
        f.write("| Pair | Overlap count |\n")
        f.write("|---|---:|\n")
        for pair in ("train_calibration", "train_test", "calibration_test"):
            f.write(f"| {pair} | {overlaps[pair]['count']} |\n")  # type: ignore[index]
        f.write("\n")
        f.write("No TIC-level host leakage is present if all overlap counts are zero.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", default="data/processed/full_dv_xml_merged.csv")
    parser.add_argument("--out-json", default="reports/validation/split_leakage_audit.json")
    parser.add_argument("--out-md", default="reports/validation/split_leakage_audit.md")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--run-date", default="2026-07-18")
    args = parser.parse_args()

    merged = pathlib.Path(args.merged)
    rows = read_csv(merged)
    train_idx, cal_idx, test_idx = split_indices(rows, args.seed)

    host_sets = {
        "train": {host_id(rows[i], i) for i in train_idx},
        "calibration": {host_id(rows[i], i) for i in cal_idx},
        "test": {host_id(rows[i], i) for i in test_idx},
    }
    host_overlaps = {
        "train_calibration": overlap(host_sets["train"], host_sets["calibration"]),
        "train_test": overlap(host_sets["train"], host_sets["test"]),
        "calibration_test": overlap(host_sets["calibration"], host_sets["test"]),
    }
    overlap_report = {
        name: {
            "count": len(values),
            "examples": values[:20],
        }
        for name, values in host_overlaps.items()
    }
    has_leakage = any(values for values in host_overlaps.values())

    strict = strict_indices(rows)
    all_strict_hosts = {host_id(rows[i], i) for i in strict}
    report: dict[str, object] = {
        "run_date": args.run_date,
        "input": str(merged),
        "seed": args.seed,
        "n_rows_total": len(rows),
        "n_strict_rows": len(strict),
        "n_strict_hosts": len(all_strict_hosts),
        "leakage_status": "FAIL" if has_leakage else "PASS",
        "host_overlaps": overlap_report,
        "splits": {
            "train": split_summary(rows, train_idx),
            "calibration": split_summary(rows, cal_idx),
            "test": split_summary(rows, test_idx),
        },
    }

    json_path = pathlib.Path(args.out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(pathlib.Path(args.out_md), report)
    print(json.dumps({
        "leakage_status": report["leakage_status"],
        "n_strict_rows": report["n_strict_rows"],
        "n_strict_hosts": report["n_strict_hosts"],
        "splits": report["splits"],
        "host_overlaps": overlap_report,
    }, indent=2, sort_keys=True))
    return 1 if has_leakage else 0


if __name__ == "__main__":
    raise SystemExit(main())
