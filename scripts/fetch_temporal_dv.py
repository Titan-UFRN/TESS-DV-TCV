#!/usr/bin/env python3
"""Fetch period-matched DV headers whose products were archived by a cutoff."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import pathlib
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from fetch_dv_subset import flatten_matching


EXOMAST = "https://exo.mast.stsci.edu/api/v0.1/dvdata/tess"
ARCHIVE = "https://archive.stsci.edu/missions/tess/tid"
SECTOR_RE = re.compile(r"^s(\d{4})-s(\d{4})$")
PRODUCT_RE = re.compile(
    r'href="([^"]+_dvr\.xml(?:\.gz)?)".*?(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})'
)


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


def get_text(url: str, timeout: int) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tess-rv-triage/0.2 (temporal validation)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def get_json(url: str, timeout: int) -> dict:
    parsed = json.loads(get_text(url, timeout))
    return parsed if isinstance(parsed, dict) else {}


def as_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
        return number if math.isfinite(number) else math.nan
    except (TypeError, ValueError):
        return math.nan


def parse_event(value: str) -> tuple[str, str, int, int] | None:
    if ":" not in value:
        return None
    sector, tce = value.split(":", 1)
    match = SECTOR_RE.match(sector)
    if not match:
        return None
    return sector, tce, int(match.group(1)), int(match.group(2))


def tce_number(value: str) -> int:
    try:
        return int(value.rsplit("_", 1)[-1])
    except ValueError:
        return 999


def tic_directory(tic: str, sector_end: int) -> str:
    digits = f"{int(tic):016d}"
    groups = "/".join(digits[index : index + 4] for index in range(0, 16, 4))
    return f"{ARCHIVE}/s{sector_end:04d}/{groups}/"


def eligible_products(
    listing: str,
    directory_url: str,
    sector_range: str,
    tic: str,
    cutoff: datetime,
) -> list[tuple[datetime, str, str]]:
    padded_tic = f"{int(tic):016d}"
    matches: list[tuple[datetime, str, str]] = []
    for filename, date_text, time_text in PRODUCT_RE.findall(listing):
        filename = html.unescape(filename)
        if f"-{sector_range}-{padded_tic}-" not in filename:
            continue
        modified = datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M")
        if modified <= cutoff:
            matches.append((modified, filename, directory_url + filename))
    return sorted(matches, reverse=True)


def empty_result(row: dict[str, str]) -> dict[str, object]:
    return {
        "tic_id": row.get("tic_id", ""),
        "toi": row.get("toi", ""),
        "temporal_role": row.get("temporal_role", ""),
        "dv_host_tce_count_total": 0,
        "dv_cutoff_eligible_event_count": 0,
        "dv_events_probed": 0,
        "dv_candidate_match": 0,
        "dv_match_reason": "",
        "dv_candidate_period_rel_error": "",
        "dv_has_tce": 0,
        "dv_tce_count": 0,
        "dv_selected_sector": "",
        "dv_selected_tce": "",
        "dv_product_filename": "",
        "dv_product_last_modified": "",
        "xml_product_url": "",
        "dv_error": "",
    }


def process_host(
    host_rows: list[dict[str, str]],
    cutoff: datetime,
    sector_max: int,
    period_rtol: float,
    timeout: int,
    max_events: int,
) -> list[dict[str, object]]:
    tic = host_rows[0]["tic_id"]
    results = [empty_result(row) for row in host_rows]
    try:
        tce_payload = get_json(f"{EXOMAST}/{tic}/tces/", timeout)
    except Exception as exc:
        for result in results:
            result["dv_match_reason"] = "tce_query_error"
            result["dv_error"] = f"{type(exc).__name__}:{exc}"
        return results

    raw_events = tce_payload.get("TCE", [])
    parsed_events = [parse_event(str(value)) for value in raw_events if value]
    events = [
        event
        for event in parsed_events
        if event is not None and 1 <= event[2] <= event[3] <= sector_max
    ]
    info_cache: dict[tuple[str, str], dict] = {}
    listing_cache: dict[int, tuple[str, str]] = {}

    for row, result in zip(host_rows, results):
        result["dv_host_tce_count_total"] = len(raw_events) if isinstance(raw_events, list) else 0
        result["dv_cutoff_eligible_event_count"] = len(events)
        if not events:
            result["dv_match_reason"] = "no_cutoff_eligible_tce"
            continue
        candidate_period = as_float(row.get("historic_period_days"))
        if not math.isfinite(candidate_period):
            result["dv_match_reason"] = "missing_candidate_period"
            continue
        signal_id = row.get("historic_signal_id", "").strip()
        preferred_tce = f"TCE_{signal_id}" if signal_id.isdigit() else ""
        ordered = sorted(
            events,
            key=lambda item: (
                item[1] != preferred_tce if preferred_tce else False,
                -(item[3] - item[2] + 1),
                -item[3],
                tce_number(item[1]),
            ),
        )
        period_match_without_product = False
        for sector, tce, _start, end in ordered[:max_events]:
            result["dv_events_probed"] = int(result["dv_events_probed"]) + 1
            cache_key = (sector, tce)
            if cache_key not in info_cache:
                query = urllib.parse.urlencode({"tce": tce, "sector": sector})
                try:
                    info_cache[cache_key] = get_json(f"{EXOMAST}/{tic}/info/?{query}", timeout)
                except Exception:
                    info_cache[cache_key] = {}
            info = info_cache[cache_key]
            data_header = info.get("DV Data Header", {}) if isinstance(info, dict) else {}
            dv_period = as_float(data_header.get("TPERIOD") if isinstance(data_header, dict) else None)
            if not math.isfinite(dv_period):
                continue
            rel_error = abs(candidate_period - dv_period) / max(abs(candidate_period), abs(dv_period))
            if rel_error > period_rtol:
                continue
            period_match_without_product = True
            if end not in listing_cache:
                directory_url = tic_directory(tic, end)
                try:
                    listing_cache[end] = (directory_url, get_text(directory_url, timeout))
                except Exception:
                    listing_cache[end] = (directory_url, "")
            directory_url, listing = listing_cache[end]
            products = eligible_products(listing, directory_url, sector, tic, cutoff)
            if not products:
                continue
            modified, filename, product_url = products[0]
            result.update(
                {
                    "dv_candidate_match": 1,
                    "dv_match_reason": "matched_cutoff_product",
                    "dv_candidate_period_rel_error": rel_error,
                    "dv_has_tce": 1,
                    "dv_tce_count": len(events),
                    "dv_selected_sector": sector,
                    "dv_selected_tce": tce,
                    "dv_product_filename": filename,
                    "dv_product_last_modified": modified.isoformat(timespec="minutes"),
                    "xml_product_url": product_url,
                }
            )
            flatten_matching("dv_data", data_header, result)
            break
        if not result["dv_candidate_match"]:
            result["dv_match_reason"] = (
                "period_match_but_no_pre_cutoff_xml" if period_match_without_product else "period_mismatch"
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="data/processed/temporal_cohort_20210617.csv")
    parser.add_argument("--out", default="data/processed/temporal_dv_20210617.csv")
    parser.add_argument("--cutoff", default="2021-06-17T23:59")
    parser.add_argument("--sector-max", type=int, default=36)
    parser.add_argument("--period-match-rtol", type=float, default=0.01)
    parser.add_argument("--max-events-per-candidate", type=int, default=30)
    parser.add_argument("--limit-hosts", type=int, default=0, help="Smoke-test limit after TIC sorting.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    rows = [
        row
        for row in read_csv(pathlib.Path(args.cohort))
        if row.get("temporal_role")
        in {"development_resolved_at_cutoff", "future_resolved_test"}
    ]
    by_host: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_host[row["tic_id"]].append(row)
    if args.limit_hosts:
        selected = sorted(by_host, key=int)[: args.limit_hosts]
        by_host = {tic: by_host[tic] for tic in selected}
    cutoff = datetime.fromisoformat(args.cutoff)
    output: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                process_host,
                host_rows,
                cutoff,
                args.sector_max,
                args.period_match_rtol,
                args.timeout,
                args.max_events_per_candidate,
            ): tic
            for tic, host_rows in by_host.items()
        }
        for index, future in enumerate(as_completed(futures), start=1):
            output.extend(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"completed hosts {index}/{len(futures)}", flush=True)
    output.sort(key=lambda row: (int(str(row["tic_id"])), float(str(row["toi"]))))
    write_rows(pathlib.Path(args.out), output)
    print(
        json.dumps(
            {
                "rows": len(output),
                "hosts": len(by_host),
                "candidate_matches": sum(row["dv_candidate_match"] == 1 for row in output),
                "xml_products": sum(bool(row["xml_product_url"]) for row in output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
