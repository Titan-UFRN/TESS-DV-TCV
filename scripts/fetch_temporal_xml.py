#!/usr/bin/env python3
"""Download and parse DVR XML products proven available by the temporal cutoff."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pathlib
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

from fetch_parse_dv_xml_subset import attrs_and_text


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_one(row: dict[str, str], timeout: int) -> dict[str, object]:
    out: dict[str, object] = {
        "tic_id": row.get("tic_id", ""),
        "toi": row.get("toi", ""),
        "dv_selected_sector": row.get("dv_selected_sector", ""),
        "dv_selected_tce": row.get("dv_selected_tce", ""),
        "xml_product_filename": row.get("dv_product_filename", ""),
        "xml_product_last_modified": row.get("dv_product_last_modified", ""),
        "xml_found": 0,
        "xml_error": "",
    }
    url = row.get("xml_product_url", "")
    if row.get("dv_candidate_match") not in {"1", "1.0"} or not url:
        out["xml_error"] = "no_cutoff_eligible_dv_product"
        return out
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "tess-rv-triage/0.2 (temporal validation)"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
        if url.endswith(".gz"):
            body = gzip.decompress(body)
        root = ET.fromstring(body)
        out["xml_found"] = 1
        out.update(attrs_and_text(root))
    except Exception as exc:
        out["xml_error"] = f"{type(exc).__name__}:{exc}"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dv", default="data/processed/temporal_dv_20210617.csv")
    parser.add_argument("--out", default="data/processed/temporal_xml_20210617.csv")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    rows = read_csv(pathlib.Path(args.dv))
    output: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(parse_one, row, args.timeout): row for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            output.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"completed XML {index}/{len(futures)}", flush=True)
    output.sort(key=lambda row: (int(str(row["tic_id"])), float(str(row["toi"]))))
    write_rows(pathlib.Path(args.out), output)
    print(
        json.dumps(
            {
                "rows": len(output),
                "xml_found": sum(row["xml_found"] == 1 for row in output),
                "xml_errors": sum(bool(row["xml_error"]) for row in output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
