#!/usr/bin/env python3
"""Download and parse DVR XML products for all DV rows, resume-safe."""

from __future__ import annotations

import argparse
import csv
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from fetch_parse_dv_xml_subset import fetch_parse_one


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def existing_tics(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row.get("tic_id", "") for row in csv.DictReader(f) if row.get("tic_id")}


def append_row(path: pathlib.Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tic_id",
        "toi",
        "strict_planet_class",
        "dv_selected_sector",
        "dv_selected_tce",
        "xml_found",
        "xml_product_filename",
        "xml_error",
        "xml_matched_elements",
        "xml_numeric_fields",
        "xml_oddeventransitdepthcomparisonstatistic_value",
        "xml_oddeventransitdepthcomparisonstatistic_significance",
        "xml_weaksecondary_maxmes",
        "xml_weaksecondary_robuststatistic",
        "xml_bootstrapresults_significance",
        "xml_bootstrapresults_transitcount",
        "xml_differenceimageresults_numberoftransits",
        "xml_differenceimageresults_numberofcadencesintransit",
        "xml_differenceimageresults_numberofcadencesoutoftransit",
        "xml_planetcandidate_maxmultipleeventsigma",
        "xml_planetcandidate_maxsingleeventsigma",
        "xml_planetcandidate_maxsesinmes",
        "xml_planetcandidate_robuststatistic",
        "xml_planetcandidate_chisquare2",
        "xml_planetcandidate_chisquaredof2",
        "xml_planetcandidate_chisquaregof",
        "xml_planetcandidate_chisquaregofdof",
        "xml_alltransitsfit_modelfitsnr",
        "xml_oddtransitsfit_modelfitsnr",
        "xml_eventransitsfit_modelfitsnr",
        "xml_trapezoidalfit_modelfitsnr",
        "xml_depthppm_value",
        "xml_depthppm_uncertainty",
        "xml_radius_value",
        "xml_radius_uncertainty",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dv", default="data/processed/dv_all.csv")
    parser.add_argument("--out", default="data/processed/dv_xml_all.csv")
    parser.add_argument("--xml-dir", default="data/raw/dv_xml_all")
    parser.add_argument("--keep-xml", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    rows = read_csv(pathlib.Path(args.dv))
    if args.limit:
        rows = rows[: args.limit]
    done = existing_tics(pathlib.Path(args.out))
    pending = [row for row in rows if row.get("tic_id") and row["tic_id"] not in done]
    print(f"rows={len(rows)} done={len(done)} pending={len(pending)}", flush=True)
    out_path = pathlib.Path(args.out)
    xml_dir = pathlib.Path(args.xml_dir)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(fetch_parse_one, row, args.timeout, xml_dir, args.keep_xml): row
            for row in pending
        }
        for i, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "tic_id": row.get("tic_id", ""),
                    "toi": row.get("toi", ""),
                    "strict_planet_class": row.get("strict_planet_class", ""),
                    "dv_selected_sector": row.get("dv_selected_sector", ""),
                    "dv_selected_tce": row.get("dv_selected_tce", ""),
                    "xml_found": 0,
                    "xml_product_filename": "",
                    "xml_error": f"{type(exc).__name__}:{exc}",
                }
            append_row(out_path, result)
            if i % 25 == 0 or i == len(pending):
                print(f"completed XML {i}/{len(pending)} latest TIC {row['tic_id']}", flush=True)
            time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
