#!/usr/bin/env python3
"""Probe MAST TIC API column names for one TIC ID."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.error
import urllib.request


ENDPOINT = "https://mast.stsci.edu/api/v0/invoke"


def call(columns: str) -> None:
    req = {
        "service": "Mast.Catalogs.Filtered.Tic.Rows",
        "format": "json",
        "params": {
            "columns": columns,
            "filters": [{"paramName": "ID", "values": ["27916356"]}],
        },
    }
    data = urllib.parse.urlencode({"request": json.dumps(req)}).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    try:
        body = urllib.request.urlopen(request, timeout=60).read().decode("utf-8", "replace")
        print("OK", columns)
        print(body[:2000])
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        print("ERR", columns, exc.code)
        print(body[:2000])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for columns in [
        "ID,mass,rad,Teff,logg,Tmag",
        "ID,Mass,rad,Teff,logg,Tmag",
        "ID,rad,Teff,logg,Tmag",
        "*",
    ]:
        call(columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
