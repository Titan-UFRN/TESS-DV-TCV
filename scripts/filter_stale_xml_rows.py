#!/usr/bin/env python3
"""Remove stale no_dv_tce XML rows when final DV metadata now has a TCE."""

from __future__ import annotations

import argparse
import csv
import pathlib


def rows(path: str) -> list[dict[str, str]]:
    with pathlib.Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dv", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dv_has_tce = {r["tic_id"] for r in rows(args.dv) if r.get("dv_has_tce") == "1"}
    xml_rows = rows(args.xml)
    keep = [
        r for r in xml_rows
        if not (r.get("tic_id") in dv_has_tce and r.get("xml_error") == "no_dv_tce")
    ]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(xml_rows[0].keys()))
        writer.writeheader()
        writer.writerows(keep)
    print("xml_rows", len(xml_rows), "keep", len(keep), "removed_stale", len(xml_rows) - len(keep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
