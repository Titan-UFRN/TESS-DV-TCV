#!/usr/bin/env python3
"""Build a leakage-audited historical-development and future-outcome cohort."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import Counter


CUTOFF_POSITIVE = {"CP", "KP"}
CUTOFF_NEGATIVE = {"EB", "V", "IS", "O", "FP", "FA"}
CUTOFF_UNRESOLVED = {"", "PC", "APC"}
CURRENT_POSITIVE = {"CP", "KP"}
CURRENT_NEGATIVE = {"EB", "V", "IS", "O", "FP", "FA"}
CURRENT_RESOLVED = CURRENT_POSITIVE | CURRENT_NEGATIVE


def read_historical(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def current_fields(row: dict[str, str]) -> tuple[str, str, str, str]:
    if "Full TOI ID" in row:
        return (
            normalized_toi(row.get("Full TOI ID", "")),
            row.get("TOI Disposition", "").strip(),
            row.get("Updated", "").strip(),
            "current_mast_toi_disposition",
        )
    return (
        normalized_toi(row.get("toi", "")),
        row.get("tfopwg_disp", "").strip(),
        row.get("rowupdate", "").strip(),
        "current_tfopwg_disposition",
    )


def normalized_toi(value: str) -> str:
    try:
        return f"{float(value):.2f}"
    except ValueError:
        return value.strip()


def finite(value: str) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else math.nan
    except (TypeError, ValueError):
        return math.nan


def sector_count(value: str) -> int:
    sectors = set()
    for item in value.replace(",", " ").split():
        try:
            sectors.add(int(item.lstrip("sS")))
        except ValueError:
            continue
    return len(sectors)


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


def feature(row: dict[str, str], source: str, target: str, out: dict[str, object]) -> None:
    value = finite(row.get(source, ""))
    out[target] = value
    out[target + "_missing"] = int(not math.isfinite(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", default="data/raw/temporal/toi_20210617.csv")
    parser.add_argument("--current", default="data/raw/temporal/toi_current_mast.csv")
    parser.add_argument("--out", default="data/processed/temporal_cohort_20210617.csv")
    parser.add_argument("--audit-json", default="reports/temporal/cohort_audit.json")
    parser.add_argument("--audit-md", default="reports/temporal/cohort_audit.md")
    args = parser.parse_args()

    historical = read_historical(pathlib.Path(args.historical))
    current_rows = read_csv(pathlib.Path(args.current))
    current = {current_fields(row)[0]: current_fields(row) for row in current_rows}

    future_test_hosts = {
        row.get("TIC", "").strip()
        for row in historical
        if row.get("TOI Disposition", "").strip() in CUTOFF_UNRESOLVED
        and normalized_toi(row.get("Full TOI ID", "")) in current
        and current[normalized_toi(row.get("Full TOI ID", ""))][1] in CURRENT_RESOLVED
    }

    output: list[dict[str, object]] = []
    for source in historical:
        toi = normalized_toi(source.get("Full TOI ID", ""))
        tic = source.get("TIC", "").strip()
        cutoff_disp = source.get("TOI Disposition", "").strip()
        _current_toi, current_disp, current_updated, current_source = current.get(
            toi, (toi, "", "", "")
        )

        if cutoff_disp in CUTOFF_UNRESOLVED and current_disp in CURRENT_RESOLVED:
            role = "future_resolved_test"
            label = int(current_disp in CURRENT_POSITIVE)
            label_source = current_source
        elif cutoff_disp in CUTOFF_POSITIVE | CUTOFF_NEGATIVE and tic in future_test_hosts:
            role = "excluded_development_host_overlap"
            label = ""
            label_source = ""
        elif cutoff_disp in CUTOFF_POSITIVE | CUTOFF_NEGATIVE:
            role = "development_resolved_at_cutoff"
            label = int(cutoff_disp in CUTOFF_POSITIVE)
            label_source = "historical_toi_disposition"
        elif cutoff_disp in CUTOFF_UNRESOLVED:
            role = "future_unresolved_or_unmapped"
            label = ""
            label_source = ""
        else:
            role = "excluded_unknown_cutoff_status"
            label = ""
            label_source = ""

        row: dict[str, object] = {
            "tic_id": tic,
            "toi": toi,
            "cutoff_date": "2021-06-17",
            "cutoff_sector_max": 36,
            "cutoff_disposition": cutoff_disp,
            "current_disposition": current_disp,
            "current_rowupdate_audit": current_updated,
            "temporal_role": role,
            "temporal_label": label,
            "temporal_label_source": label_source,
            "historic_pipeline": source.get("Source Pipeline", "").strip(),
            "historic_pipeline_spoc": int(source.get("Source Pipeline", "").strip().lower() == "spoc"),
            "historic_sector_count": sector_count(source.get("Sectors", "")),
            "historic_sectors": source.get("Sectors", "").strip(),
            "historic_signal_id": source.get("Signal ID", "").strip(),
            "historic_alerted": source.get("Alerted", "").strip(),
            "historic_updated": source.get("Updated", "").strip(),
        }
        for source_field, target_field in [
            ("Orbital Period Value", "historic_period_days"),
            ("Epoch Value", "historic_epoch_btjd"),
            ("Transit Duration Value", "historic_transit_duration_hours"),
            ("Transit Depth Value", "historic_transit_depth_ppm"),
            ("Planet Radius Value", "historic_planet_radius_rearth"),
            ("Star Radius Value", "historic_stellar_radius_rsun"),
            ("Effective Temperature Value", "historic_stellar_teff"),
            ("Surface Gravity Value", "historic_stellar_logg"),
            ("TMag Value", "historic_tmag"),
            ("Signal-to-noise", "historic_signal_to_noise"),
            ("Centroid Offset", "historic_centroid_offset"),
        ]:
            feature(source, source_field, target_field, row)
        output.append(row)

    write_rows(pathlib.Path(args.out), output)
    development = [row for row in output if row["temporal_role"] == "development_resolved_at_cutoff"]
    test = [row for row in output if row["temporal_role"] == "future_resolved_test"]
    development_hosts = {str(row["tic_id"]) for row in development}
    test_hosts = {str(row["tic_id"]) for row in test}
    audit = {
        "cutoff_date": "2021-06-17",
        "cutoff_sector_max": 36,
        "historical_rows": len(historical),
        "current_rows": len(current_rows),
        "current_outcome_disposition_system": (
            "TESS Science Office TOI Disposition"
            if current_rows and "Full TOI ID" in current_rows[0]
            else "TFOPWG Disposition"
        ),
        "joined_rows": sum(normalized_toi(row.get("Full TOI ID", "")) in current for row in historical),
        "role_counts": dict(Counter(str(row["temporal_role"]) for row in output)),
        "cutoff_disposition_counts": dict(Counter(str(row["cutoff_disposition"]) for row in output)),
        "development_rows": len(development),
        "development_hosts": len(development_hosts),
        "development_positive": sum(row["temporal_label"] == 1 for row in development),
        "development_negative": sum(row["temporal_label"] == 0 for row in development),
        "future_test_rows": len(test),
        "future_test_hosts": len(test_hosts),
        "future_test_positive": sum(row["temporal_label"] == 1 for row in test),
        "future_test_negative": sum(row["temporal_label"] == 0 for row in test),
        "host_overlap": len(development_hosts & test_hosts),
        "duplicate_candidate_keys": len(output) - len({(row["tic_id"], row["toi"]) for row in output}),
        "current_numeric_features_used": [],
    }
    if audit["host_overlap"] or audit["duplicate_candidate_keys"]:
        raise RuntimeError(f"temporal cohort integrity failure: {audit}")

    audit_path = pathlib.Path(args.audit_json)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    md = [
        "# Temporal Cohort Audit",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| Historical snapshot rows | {audit['historical_rows']} |",
        f"| Development rows / hosts | {audit['development_rows']} / {audit['development_hosts']} |",
        f"| Development positive / negative | {audit['development_positive']} / {audit['development_negative']} |",
        f"| Future-resolved test rows / hosts | {audit['future_test_rows']} / {audit['future_test_hosts']} |",
        f"| Future test positive / negative | {audit['future_test_positive']} / {audit['future_test_negative']} |",
        f"| Development-test host overlap | {audit['host_overlap']} |",
        f"| Duplicate candidate keys | {audit['duplicate_candidate_keys']} |",
        "",
        "The fixed test cohort contains candidates unresolved in the archived 2021-06-17 TOI catalog and resolved in the current TFOPWG snapshot. Current numeric parameters are not used as features.",
    ]
    pathlib.Path(args.audit_md).write_text("\n".join(md) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
