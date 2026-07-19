#!/usr/bin/env python3
"""Download and parse TESS DVR XML products for the DV subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


EXOMAST = "https://exo.mast.stsci.edu/api/v0.1/dvdata/tess"
MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file"

FIELD_PATTERNS = [
    "odd",
    "even",
    "secondary",
    "weaksecondary",
    "centroid",
    "differenceimage",
    "difference",
    "bootstrap",
    "ghost",
    "rollingband",
    "limb",
    "trapezoid",
    "planet",
    "transit",
    "mes",
    "snr",
    "depth",
    "duration",
    "period",
    "epoch",
    "radius",
    "impact",
    "prad",
]


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_json(url: str, timeout: int) -> dict:
    body = urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8", "replace")
    parsed = json.loads(body)
    return parsed if isinstance(parsed, dict) else {}


def info_for(row: dict[str, str], timeout: int) -> dict:
    tic = row["tic_id"]
    sector = row.get("dv_selected_sector", "")
    tce = row.get("dv_selected_tce", "")
    if not sector or not tce:
        return {}
    url = f"{EXOMAST}/{tic}/info/?{urllib.parse.urlencode({'tce': tce, 'sector': sector})}"
    return get_json(url, timeout)


def xml_filename_from_info(info: dict) -> str:
    primary = info.get("DV Primary Header") or {}
    dv_file = str(primary.get("dvFileName") or primary.get("DVFILENAME") or "")
    name = pathlib.PurePosixPath(dv_file).name
    if name.endswith("_dvt.fits"):
        return name.replace("_dvt.fits", "_dvr.xml")
    resfile = str(primary.get("RESFILE") or "")
    if resfile.endswith(".xml.gz"):
        return pathlib.PurePosixPath(resfile[:-3]).name
    if resfile.endswith(".xml"):
        return pathlib.PurePosixPath(resfile).name
    return ""


def download_product(filename: str, timeout: int) -> bytes:
    url = MAST_DOWNLOAD + "?" + urllib.parse.urlencode({"uri": "mast:TESS/product/" + filename})
    return urllib.request.urlopen(url, timeout=timeout).read()


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def text_float(text: str | None) -> float:
    if text is None:
        return math.nan
    try:
        value = float(text.strip())
        return value if math.isfinite(value) else math.nan
    except ValueError:
        return math.nan


def attrs_and_text(root: ET.Element) -> dict[str, object]:
    """Extract numeric/string diagnostics from likely relevant XML elements."""
    out: dict[str, object] = {}
    matched_elements = 0
    numeric_fields = 0
    for elem in root.iter():
        tag = strip_ns(elem.tag)
        tag_l = tag.lower()
        path_hit = any(pat in tag_l for pat in FIELD_PATTERNS)
        attr_hit = any(any(pat in str(k).lower() for pat in FIELD_PATTERNS) for k in elem.attrib)
        if not path_hit and not attr_hit:
            continue
        matched_elements += 1
        base = normalize_key("xml_" + tag)
        val = text_float(elem.text)
        if math.isfinite(val):
            out.setdefault(base, val)
            numeric_fields += 1
        for key, value in elem.attrib.items():
            out_key = normalize_key(f"{base}_{key}")
            fval = text_float(value)
            if math.isfinite(fval):
                out.setdefault(out_key, fval)
                numeric_fields += 1
            elif len(str(value)) < 80:
                out.setdefault(out_key, value)
    out["xml_matched_elements"] = matched_elements
    out["xml_numeric_fields"] = numeric_fields
    return out


def normalize_key(key: str) -> str:
    key = re.sub(r"[^0-9A-Za-z_]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_").lower()
    return key[:120]


def fetch_parse_one(row: dict[str, str], timeout: int, xml_dir: pathlib.Path, keep_xml: bool) -> dict[str, object]:
    out: dict[str, object] = {
        "tic_id": row.get("tic_id", ""),
        "toi": row.get("toi", ""),
        "strict_planet_class": row.get("strict_planet_class", ""),
        "dv_subset_group": row.get("dv_subset_group", ""),
        "dv_selected_sector": row.get("dv_selected_sector", ""),
        "dv_selected_tce": row.get("dv_selected_tce", ""),
        "xml_found": 0,
        "xml_product_filename": "",
        "xml_error": "",
    }
    if row.get("dv_has_tce") not in {"1", "1.0"}:
        out["xml_error"] = "no_dv_tce"
        return out
    try:
        info = info_for(row, timeout)
        filename = xml_filename_from_info(info)
        out["xml_product_filename"] = filename
        if not filename:
            out["xml_error"] = "no_xml_filename"
            return out
        xml_bytes = download_product(filename, timeout)
        if keep_xml:
            xml_dir.mkdir(parents=True, exist_ok=True)
            (xml_dir / filename).write_bytes(xml_bytes)
        root = ET.fromstring(xml_bytes)
        out["xml_found"] = 1
        out.update(attrs_and_text(root))
    except urllib.error.HTTPError as exc:
        out["xml_error"] = f"http_{exc.code}"
    except ET.ParseError as exc:
        out["xml_error"] = f"parse_error:{exc}"
    except Exception as exc:
        out["xml_error"] = f"{type(exc).__name__}:{exc}"
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
    parser.add_argument("--dv", default="data/processed/dv_subset.csv")
    parser.add_argument("--out", default="data/processed/dv_xml_features.csv")
    parser.add_argument("--xml-dir", default="data/raw/dv_xml")
    parser.add_argument("--keep-xml", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    rows = read_csv(pathlib.Path(args.dv))
    out_rows = []
    for i, row in enumerate(rows, start=1):
        print(f"fetching XML {i}/{len(rows)} TIC {row.get('tic_id')}", flush=True)
        out_rows.append(fetch_parse_one(row, args.timeout, pathlib.Path(args.xml_dir), args.keep_xml))
        time.sleep(args.sleep)
    write_rows(pathlib.Path(args.out), out_rows)
    print(f"wrote {len(out_rows)} rows to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
