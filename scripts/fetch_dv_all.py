#!/usr/bin/env python3
"""Fetch lightweight Exo.MAST DV metadata for all candidates, resume-safe."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from fetch_dv_subset import choose_tce, flatten_matching


BASE = "https://exo.mast.stsci.edu/api/v0.1/dvdata/tess"


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def existing_tics(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row.get("tic_id", "") for row in csv.DictReader(f) if row.get("tic_id")}


def get_json(url: str, timeout: int) -> object:
    try:
        body = urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}", "body": exc.read().decode("utf-8", "replace")[:500]}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"error": "json_decode", "body": body[:500]}


def fetch_one(row: dict[str, str], timeout: int) -> dict[str, object]:
    tic = row["tic_id"]
    out: dict[str, object] = {
        "tic_id": tic,
        "toi": row.get("toi", ""),
        "strict_planet_class": row.get("strict_planet_class", ""),
        "tfopwg_disp": row.get("tfopwg_disp", ""),
        "dv_has_tce": 0,
        "dv_tce_count": 0,
        "dv_selected_sector": "",
        "dv_selected_tce": "",
        "dv_error": "",
    }
    tces_obj = get_json(f"{BASE}/{tic}/tces/", timeout)
    if isinstance(tces_obj, dict) and "error" in tces_obj:
        out["dv_error"] = str(tces_obj["error"])
        return out
    tces = tces_obj.get("TCE", []) if isinstance(tces_obj, dict) else []
    if not isinstance(tces, list) or not tces:
        return out
    out["dv_has_tce"] = 1
    out["dv_tce_count"] = len(tces)
    sector, tce = choose_tce([str(x) for x in tces])
    out["dv_selected_sector"] = sector
    out["dv_selected_tce"] = tce
    if not sector or not tce:
        return out
    info_url = f"{BASE}/{tic}/info/?{urllib.parse.urlencode({'tce': tce, 'sector': sector})}"
    info_obj = get_json(info_url, timeout)
    if isinstance(info_obj, dict) and "error" in info_obj:
        out["dv_error"] = str(info_obj["error"])
        return out
    if isinstance(info_obj, dict):
        flatten_matching("dv_primary", info_obj.get("DV Primary Header"), out)
        flatten_matching("dv_data", info_obj.get("DV Data Header"), out)
    return out


def append_row(path: pathlib.Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tic_id",
        "toi",
        "strict_planet_class",
        "tfopwg_disp",
        "dv_has_tce",
        "dv_tce_count",
        "dv_selected_sector",
        "dv_selected_tce",
        "dv_error",
        "dv_primary_logg",
        "dv_primary_radius",
        "dv_primary_teff",
        "dv_primary_tessmag",
        "dv_data_impact",
        "dv_data_indur",
        "dv_data_maxmes",
        "dv_data_pradius",
        "dv_data_tdepth",
        "dv_data_tdur",
        "dv_data_tepoch",
        "dv_data_tperiod",
        "dv_data_tsnr",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--out", default="data/processed/dv_all.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    rows = read_csv(pathlib.Path(args.candidates))
    if args.limit:
        rows = rows[: args.limit]
    done = existing_tics(pathlib.Path(args.out))
    pending = [row for row in rows if row.get("tic_id") and row["tic_id"] not in done]
    print(f"rows={len(rows)} done={len(done)} pending={len(pending)}", flush=True)
    out_path = pathlib.Path(args.out)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch_one, row, args.timeout): row for row in pending}
        for i, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "tic_id": row.get("tic_id", ""),
                    "toi": row.get("toi", ""),
                    "strict_planet_class": row.get("strict_planet_class", ""),
                    "tfopwg_disp": row.get("tfopwg_disp", ""),
                    "dv_has_tce": 0,
                    "dv_tce_count": 0,
                    "dv_selected_sector": "",
                    "dv_selected_tce": "",
                    "dv_error": f"{type(exc).__name__}:{exc}",
                }
            append_row(out_path, result)
            if i % 25 == 0 or i == len(pending):
                print(f"completed {i}/{len(pending)} latest TIC {row['tic_id']}", flush=True)
            time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
