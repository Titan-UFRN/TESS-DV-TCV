#!/usr/bin/env python3
"""Probe public URL patterns for TESS DV XML products."""

from __future__ import annotations

import argparse
import urllib.request
import urllib.parse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    urls = [
        "https://archive.stsci.edu/missions/tess/tid/s0032/0000/0000/7722/8304/tess2020324010417-s0032-s0032-0000000077228304-00419_dvr.xml.gz",
        "https://archive.stsci.edu/missions/tess/public/tid/s0032/0000/0000/7722/8304/tess2020324010417-s0032-s0032-0000000077228304-00419_dvr.xml.gz",
    ]
    filename = "tess2020324010417-s0032-s0032-0000000077228304-00419_dvr.xml.gz"
    for name in [filename, filename[:-3]]:
        urls.append(
            "https://mast.stsci.edu/api/v0.1/Download/file?"
            + urllib.parse.urlencode({"uri": "mast:TESS/product/" + name})
        )
    for url in urls:
        print(url)
        try:
            resp = urllib.request.urlopen(url, timeout=60)
            print(resp.status, resp.headers.get("content-length"), resp.read(20))
        except Exception as exc:
            print(type(exc).__name__, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
