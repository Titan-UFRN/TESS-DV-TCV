#!/usr/bin/env python3
"""Probe Exo.MAST TESS DV endpoints for a few TIC IDs."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def get(url: str) -> str:
    try:
        return urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:1000]}"
    except Exception as exc:  # pragma: no cover - probe script
        return f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for tic in [77228304, 82362447, 425163745, 22529346, 350132371]:
        print("===", tic)
        tces_url = f"https://exo.mast.stsci.edu/api/v0.1/dvdata/tess/{tic}/tces/"
        body = get(tces_url)
        print("TCES", body[:1200])
        try:
            tces = json.loads(body)
        except json.JSONDecodeError:
            continue
        tce_list = tces.get("TCE", []) if isinstance(tces, dict) else tces
        if not isinstance(tce_list, list) or not tce_list:
            continue
        tce = tce_list[0]
        print("FIRST_TCE", tce)
        sector, tce_id = str(tce).split(":", 1) if ":" in str(tce) else ("", str(tce))
        for url in [
            f"https://exo.mast.stsci.edu/api/v0.1/dvdata/tess/{tic}/info/",
            f"https://exo.mast.stsci.edu/api/v0.1/dvdata/tess/{tic}/info/?list_tce=true",
            f"https://exo.mast.stsci.edu/api/v0.1/dvdata/tess/{tic}/info/?tce={tce_id}&sector={sector}",
            f"https://exo.mast.stsci.edu/api/v0.1/dvdata/tess/{tic}/table/?tce={tce_id}&sector={sector}",
        ]:
            print("URL", url)
            print(get(url)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
