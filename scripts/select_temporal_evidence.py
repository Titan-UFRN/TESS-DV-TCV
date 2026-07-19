#!/usr/bin/env python3
"""Select candidate-keyed temporal evidence for a revised outcome cohort."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib


ROLES = {"development_resolved_at_cutoff", "future_resolved_test"}


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: pathlib.Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
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


def combine(paths: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for value in paths:
        for row in read_csv(pathlib.Path(value)):
            output[key(row)] = row
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--source-dv", nargs="+", required=True)
    parser.add_argument("--source-xml", nargs="+", required=True)
    parser.add_argument("--out-dv", required=True)
    parser.add_argument("--out-xml", required=True)
    parser.add_argument("--missing-cohort", required=True)
    args = parser.parse_args()

    all_cohort = read_csv(pathlib.Path(args.cohort))
    cohort = [row for row in all_cohort if row.get("temporal_role") in ROLES]
    target = {key(row) for row in cohort}
    dv = combine(args.source_dv)
    xml = combine(args.source_xml)
    selected_dv = [dv[item] for item in sorted(target) if item in dv]
    selected_xml = [xml[item] for item in sorted(target) if item in xml]
    missing = [row for row in all_cohort if key(row) in target - (set(dv) & set(xml))]

    write_rows(pathlib.Path(args.out_dv), selected_dv)
    write_rows(pathlib.Path(args.out_xml), selected_xml)
    write_rows(
        pathlib.Path(args.missing_cohort),
        missing,
        fields=list(all_cohort[0]) if all_cohort else [],
    )
    report = {
        "target_rows": len(target),
        "selected_dv_rows": len(selected_dv),
        "selected_xml_rows": len(selected_xml),
        "missing_rows": len(missing),
        "missing_keys": [{"tic_id": tic, "toi": toi} for tic, toi in sorted(target - (set(dv) & set(xml)))],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
