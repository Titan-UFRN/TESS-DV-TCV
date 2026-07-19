#!/usr/bin/env python3
"""Deduplicate DV metadata rows by TIC, preferring rows with a TCE."""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with pathlib.Path(args.infile).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_tic: dict[str, dict[str, str]] = {}
    for row in rows:
        tic = row.get("tic_id", "")
        if not tic:
            continue
        current = by_tic.get(tic)
        if current is None or (current.get("dv_has_tce") != "1" and row.get("dv_has_tce") == "1"):
            by_tic[tic] = row

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(by_tic.values())

    print("in", len(rows), "out", len(by_tic))
    print("dv_has_tce", collections.Counter(r.get("dv_has_tce") for r in by_tic.values()))
    print("strict_planet_class", collections.Counter(r.get("strict_planet_class") for r in by_tic.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
