#!/usr/bin/env python3
"""Probe MAST CAOM products for TESS DV XML files."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request


ENDPOINT = "https://mast.stsci.edu/api/v0/invoke"


def mast(request: dict) -> dict:
    data = urllib.parse.urlencode({"request": json.dumps(request)}).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    body = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    return json.loads(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for service, params in [
        (
            "Mast.Caom.Filtered",
            {
                "columns": "obsid,obs_id,target_name,dataproduct_type,obs_collection,filters,project",
                "filters": [
                    {"paramName": "obs_collection", "values": ["TESS"]},
                    {"paramName": "target_name", "values": ["TIC 77228304"]},
                ],
            },
        ),
        (
            "Mast.Caom.Filtered",
            {
                "columns": "*",
                "filters": [
                    {"paramName": "obs_collection", "values": ["TESS"]},
                    {"paramName": "target_name", "values": ["77228304"]},
                ],
            },
        ),
    ]:
        print("SERVICE", service, params)
        try:
            out = mast({"service": service, "format": "json", "params": params})
            print(json.dumps(out, indent=2)[:4000])
        except Exception as exc:
            print(type(exc).__name__, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
