#!/usr/bin/env python3
"""Fetch TIC stellar parameters from MAST for TOI TIC IDs."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request


ENDPOINT = "https://mast.stsci.edu/api/v0/invoke"
COLUMNS = "ID,mass,e_mass,rad,e_rad,Teff,e_Teff,logg,e_logg,Tmag,GAIAmag,GAIA,lumclass,contratio,numcont"


def read_tic_ids(toi_csv: pathlib.Path) -> list[str]:
    with toi_csv.open(newline="", encoding="utf-8-sig") as f:
        ids = []
        for row in csv.DictReader(f):
            tid = "".join(ch for ch in (row.get("tid") or "") if ch.isdigit())
            if tid:
                ids.append(tid)
    return sorted(set(ids), key=int)


def mast_call(ids: list[str], timeout: int) -> list[dict[str, object]]:
    req = {
        "service": "Mast.Catalogs.Filtered.Tic.Rows",
        "format": "json",
        "params": {
            "columns": COLUMNS,
            "filters": [{"paramName": "ID", "values": ids}],
        },
    }
    data = urllib.parse.urlencode({"request": json.dumps(req)}).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    try:
        body = urllib.request.urlopen(request, timeout=timeout).read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", "replace")
        raise RuntimeError(err[:2000]) from exc
    parsed = json.loads(body)
    if parsed.get("status") not in {"COMPLETE", "EXECUTING"}:
        raise RuntimeError(body[:2000])
    return parsed.get("data", [])


def write_rows(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ID",
        "mass",
        "e_mass",
        "rad",
        "e_rad",
        "Teff",
        "e_Teff",
        "logg",
        "e_logg",
        "Tmag",
        "GAIAmag",
        "GAIA",
        "lumclass",
        "contratio",
        "numcont",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toi", default="data/raw/toi.csv")
    parser.add_argument("--out", default="data/raw/tic_stellar.csv")
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    ids = read_tic_ids(pathlib.Path(args.toi))
    all_rows: list[dict[str, object]] = []
    for start in range(0, len(ids), args.chunk_size):
        chunk = ids[start : start + args.chunk_size]
        print(f"fetching TIC {start + 1}-{start + len(chunk)} / {len(ids)}", flush=True)
        rows = mast_call(chunk, args.timeout)
        all_rows.extend(rows)
        time.sleep(args.sleep)

    write_rows(pathlib.Path(args.out), all_rows)
    print(f"wrote {len(all_rows)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
