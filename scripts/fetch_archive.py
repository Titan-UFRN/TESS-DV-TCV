#!/usr/bin/env python3
"""Fetch compact NASA Exoplanet Archive TAP snapshots for the first experiment."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.parse
import urllib.request
import urllib.error


TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

QUERIES = {
    "toi": """
        select
            tid, toi, tfopwg_disp, rastr, ra, decstr, dec,
            st_tmag, st_teff, st_logg, st_rad,
            pl_orbper, pl_trandep, pl_trandurh, pl_rade,
            pl_insol, pl_eqt
        from toi
    """,
    "ps": """
        select
            pl_name, hostname, tic_id, default_flag, discoverymethod,
            disc_facility, sy_snum, sy_pnum, tran_flag, rv_flag,
            pl_orbper, pl_rade, pl_bmasse, pl_bmassj, pl_masse,
            pl_radj, pl_orbsmax, st_teff, st_rad, st_mass,
            sy_tmag, sy_vmag, sy_gaiamag
        from ps
        where default_flag = 1
    """,
    "pscomppars": """
        select
            pl_name, hostname, tic_id, discoverymethod, disc_facility,
            sy_snum, sy_pnum, tran_flag, rv_flag,
            pl_orbper, pl_rade, pl_bmasse, pl_bmassj,
            pl_orbsmax, st_teff, st_rad, st_mass,
            sy_tmag, sy_vmag, sy_gaiamag
        from pscomppars
    """,
}


def normalize_sql(sql: str) -> str:
    return " ".join(line.strip() for line in sql.strip().splitlines())


def fetch_csv(sql: str, timeout: int) -> bytes:
    params = urllib.parse.urlencode({"query": normalize_sql(sql), "format": "csv"})
    url = f"{TAP_SYNC}?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "tess-rv-triage/0.1 (research prototype)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(body[:4000].decode("utf-8", "replace")) from exc
    if body[:100].lower().startswith(b"error") or b"<INFO name=\"QUERY_STATUS\" value=\"ERROR\"" in body[:1000]:
        raise RuntimeError(body[:2000].decode("utf-8", "replace"))
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for name, sql in QUERIES.items():
        path = outdir / f"{name}.csv"
        print(f"fetching {name} -> {path}", flush=True)
        body = fetch_csv(sql, args.timeout)
        path.write_bytes(body)
        rows = max(0, body.count(b"\n") - 1)
        manifest.append(f"{name},{path},{len(body)},{rows},{normalize_sql(sql)}")
        time.sleep(args.sleep)

    (outdir / "manifest.csv").write_text(
        "table,path,bytes,approx_rows,query\n" + "\n".join(manifest) + "\n",
        encoding="utf-8",
    )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
