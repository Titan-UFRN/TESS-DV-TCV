#!/usr/bin/env python3
"""Audit candidate identity and candidate-to-DV/XML association after merging."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import Counter, defaultdict


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def key(row: dict[str, str]) -> tuple[str, str]:
    return (str(row.get("tic_id", "")).strip(), str(row.get("toi", "")).strip())


def duplicate_summary(rows: list[dict[str, str]]) -> tuple[int, list[dict[str, object]]]:
    counts = Counter(key(row) for row in rows)
    duplicates = [
        {"tic_id": item[0], "toi": item[1], "count": count}
        for item, count in counts.most_common()
        if count > 1
    ]
    return sum(int(item["count"]) - 1 for item in duplicates), duplicates[:25]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--merged", default="data/processed/full_dv_xml_merged.csv")
    parser.add_argument("--out-json", default="reports/validation/candidate_integrity_audit.json")
    parser.add_argument("--out-md", default="reports/validation/candidate_integrity_audit.md")
    args = parser.parse_args()

    candidates = read_csv(pathlib.Path(args.candidates))
    merged = read_csv(pathlib.Path(args.merged))
    candidate_by_key = {key(row): row for row in candidates}
    merged_by_key = {key(row): row for row in merged}
    candidate_fields = list(candidates[0]) if candidates else []

    candidate_duplicate_rows, candidate_duplicate_examples = duplicate_summary(candidates)
    merged_duplicate_rows, merged_duplicate_examples = duplicate_summary(merged)
    missing_keys = sorted(set(candidate_by_key) - set(merged_by_key))
    extra_keys = sorted(set(merged_by_key) - set(candidate_by_key))

    mismatch_counts: Counter[str] = Counter()
    mismatch_examples: list[dict[str, object]] = []
    for item_key in sorted(set(candidate_by_key) & set(merged_by_key)):
        source = candidate_by_key[item_key]
        result = merged_by_key[item_key]
        differing = [field for field in candidate_fields if source.get(field, "") != result.get(field, "")]
        for field in differing:
            mismatch_counts[field] += 1
        if differing and len(mismatch_examples) < 25:
            mismatch_examples.append({"tic_id": item_key[0], "toi": item_key[1], "fields": differing})

    matched_by_tic: dict[str, list[str]] = defaultdict(list)
    for row in merged:
        if row.get("dv_candidate_match") in {"1", "1.0"}:
            matched_by_tic[row.get("tic_id", "")].append(row.get("toi", ""))
    multiply_matched = [
        {"tic_id": tic, "tois": tois, "count": len(tois)}
        for tic, tois in sorted(matched_by_tic.items())
        if len(tois) > 1
    ]

    status = "PASS"
    if (
        len(candidates) != len(merged)
        or candidate_duplicate_rows
        or merged_duplicate_rows
        or missing_keys
        or extra_keys
        or mismatch_counts
        or multiply_matched
    ):
        status = "FAIL"

    report: dict[str, object] = {
        "status": status,
        "candidate_rows": len(candidates),
        "merged_rows": len(merged),
        "candidate_unique_keys": len(candidate_by_key),
        "merged_unique_keys": len(merged_by_key),
        "candidate_duplicate_rows": candidate_duplicate_rows,
        "merged_duplicate_rows": merged_duplicate_rows,
        "candidate_duplicate_examples": candidate_duplicate_examples,
        "merged_duplicate_examples": merged_duplicate_examples,
        "missing_key_count": len(missing_keys),
        "extra_key_count": len(extra_keys),
        "missing_key_examples": [{"tic_id": tic, "toi": toi} for tic, toi in missing_keys[:25]],
        "extra_key_examples": [{"tic_id": tic, "toi": toi} for tic, toi in extra_keys[:25]],
        "identity_field_mismatch_counts": dict(mismatch_counts),
        "identity_field_mismatch_examples": mismatch_examples,
        "dv_candidate_matches": sum(row.get("dv_candidate_match") in {"1", "1.0"} for row in merged),
        "xml_candidate_matches": sum(row.get("xml_candidate_match") in {"1", "1.0"} for row in merged),
        "dv_match_reason_counts": dict(Counter(row.get("dv_match_reason", "") for row in merged)),
        "multiply_matched_tic_count": len(multiply_matched),
        "multiply_matched_tic_examples": multiply_matched[:25],
    }

    out_json = pathlib.Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_md = pathlib.Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "\n".join(
            [
                "# Candidate Integrity Audit",
                "",
                f"Status: **{status}**",
                "",
                "| Check | Value |",
                "|---|---:|",
                f"| Candidate rows | {len(candidates)} |",
                f"| Merged rows | {len(merged)} |",
                f"| Candidate duplicate rows | {candidate_duplicate_rows} |",
                f"| Merged duplicate rows | {merged_duplicate_rows} |",
                f"| Missing candidate keys | {len(missing_keys)} |",
                f"| Extra merged keys | {len(extra_keys)} |",
                f"| Identity fields with mismatches | {len(mismatch_counts)} |",
                f"| Candidate-matched DV products | {report['dv_candidate_matches']} |",
                f"| Candidate-matched XML products | {report['xml_candidate_matches']} |",
                f"| TICs matched to more than one TOI | {len(multiply_matched)} |",
                "",
                "The audit fails if candidate identity changes during the merge, if a candidate key is duplicated or lost, or if one selected host product is assigned to more than one candidate.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
