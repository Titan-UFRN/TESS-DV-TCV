#!/usr/bin/env python3
"""Fetch lightweight Exo.MAST DV metadata for a ranked subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request


BASE = "https://exo.mast.stsci.edu/api/v0.1/dvdata/tess"


KEY_PATTERNS = [
    "ODD",
    "EVEN",
    "SECON",
    "CENT",
    "DIFF",
    "DEPTH",
    "DUR",
    "PERIOD",
    "EPOCH",
    "MES",
    "SNR",
    "PRADIUS",
    "RADIUS",
    "IMPACT",
    "LOGG",
    "TEFF",
    "TESSMAG",
]


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def numeric(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else -math.inf
    except ValueError:
        return -math.inf


def select_subset(candidates: list[dict[str, str]], per_group: int) -> list[dict[str, str]]:
    groups = {
        "fp_expected_k": [
            r for r in candidates if r.get("strict_planet_class") == "false_positive_or_false_alarm"
        ],
        "confirmed_expected_k": [
            r for r in candidates if r.get("strict_planet_class") == "known_or_confirmed_planet"
        ],
        "candidate_expected_k": [
            r for r in candidates if r.get("strict_planet_class") in {"planet_candidate", "ambiguous_planet_candidate"}
        ],
    }
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for group, rows in groups.items():
        ranked = sorted(rows, key=lambda r: numeric(r, "score_expected_k"), reverse=True)
        for row in ranked[:per_group]:
            tic = row.get("tic_id", "")
            if not tic or tic in seen:
                continue
            out = dict(row)
            out["dv_subset_group"] = group
            selected.append(out)
            seen.add(tic)
    return selected


def get_json(url: str, timeout: int) -> object:
    try:
        body = urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}", "body": exc.read().decode("utf-8", "replace")[:500]}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"error": "json_decode", "body": body[:500]}


def choose_tce(tces: list[str]) -> tuple[str, str]:
    """Prefer a multi-sector TCE_1 if available."""
    parsed = []
    for item in tces:
        if ":" not in item:
            continue
        sector, tce = item.split(":", 1)
        width = sector_width(sector)
        parsed.append((width, sector, tce))
    if not parsed:
        return "", ""
    parsed.sort(key=lambda x: (x[2] != "TCE_1", -x[0], x[1]))
    _, sector, tce = parsed[0]
    return sector, tce


def sector_width(sector: str) -> int:
    nums = [int(part[1:]) for part in sector.split("-") if part.startswith("s") and part[1:].isdigit()]
    if len(nums) == 2:
        return nums[1] - nums[0] + 1
    return 1 if nums else 0


def flatten_matching(prefix: str, obj: object, out: dict[str, object]) -> None:
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        ukey = str(key).upper()
        if any(pattern in ukey for pattern in KEY_PATTERNS):
            safe_key = f"{prefix}_{ukey}".lower().replace(" ", "_")
            if isinstance(value, (str, int, float)) or value is None:
                out[safe_key] = value


def fetch_one(row: dict[str, str], timeout: int) -> dict[str, object]:
    tic = row["tic_id"]
    out: dict[str, object] = {
        "tic_id": tic,
        "toi": row.get("toi", ""),
        "strict_planet_class": row.get("strict_planet_class", ""),
        "dv_subset_group": row.get("dv_subset_group", ""),
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


def write_rows(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/processed/candidates.csv")
    parser.add_argument("--out", default="data/processed/dv_subset.csv")
    parser.add_argument("--per-group", type=int, default=80)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    candidates = read_csv(pathlib.Path(args.candidates))
    subset = select_subset(candidates, args.per_group)
    rows = []
    for i, row in enumerate(subset, start=1):
        print(f"fetching DV {i}/{len(subset)} TIC {row['tic_id']}", flush=True)
        rows.append(fetch_one(row, args.timeout))
        time.sleep(args.sleep)
    write_rows(pathlib.Path(args.out), rows)
    print(f"wrote {len(rows)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
