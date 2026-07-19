#!/usr/bin/env python3
"""Fetch the fixed historical TOI catalog and a dated current outcome snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import urllib.request
from datetime import datetime, timezone


HISTORICAL_URL = (
    "https://archive.stsci.edu/missions/tess/catalogs/toi/previous/"
    "tess20210617_toi-catalog.csv"
)
CURRENT_URL = "https://archive.stsci.edu/missions/tess/catalogs/toi/tois.csv"


def fetch(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tess-dv-tcv/1.0 (temporal validation)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def catalog_date(body: bytes) -> str:
    match = re.search(br"^# File Created:\s*([^\r\n]+)", body, re.MULTILINE)
    return match.group(1).decode("ascii", "replace").strip() if match else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", default="data/raw/temporal/toi_20210617.csv")
    parser.add_argument("--current", default="data/raw/temporal/toi_current_mast.csv")
    parser.add_argument("--provenance", default="data/raw/temporal/provenance.json")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    historical_body = fetch(HISTORICAL_URL, args.timeout)
    current_body = fetch(CURRENT_URL, args.timeout)
    if b"File Created: 2021-06-17" not in historical_body[:500]:
        raise RuntimeError("historical catalog does not contain the expected cutoff header")
    if b"Full TOI ID,TOI Disposition" not in current_body[:2000]:
        raise RuntimeError("current TOI response has an unexpected schema")

    historical_path = pathlib.Path(args.historical)
    current_path = pathlib.Path(args.current)
    historical_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.parent.mkdir(parents=True, exist_ok=True)
    historical_path.write_bytes(historical_body)
    current_path.write_bytes(current_body)

    provenance = {
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "historical_cutoff": "2021-06-17",
        "historical_sector_range": "S0001-S0036",
        "historical_url": HISTORICAL_URL,
        "historical_path": str(historical_path),
        "historical_bytes": len(historical_body),
        "historical_sha256": digest(historical_body),
        "current_outcome_catalog_date": catalog_date(current_body),
        "current_outcome_disposition_system": "TESS Science Office TOI Disposition",
        "current_outcome_url": CURRENT_URL,
        "current_outcome_path": str(current_path),
        "current_outcome_bytes": len(current_body),
        "current_outcome_sha256": digest(current_body),
    }
    provenance_path = pathlib.Path(args.provenance)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
