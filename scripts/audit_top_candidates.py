#!/usr/bin/env python3
"""Audit top unresolved TESS candidates from the triage ranking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import Counter


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


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


def fnum(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
        return out if math.isfinite(out) else math.nan
    except (TypeError, ValueError):
        return math.nan


def yn(value: str) -> bool:
    return value in {"1", "1.0", "true", "True", "yes"}


def risk_notes(row: dict[str, str]) -> list[str]:
    notes: list[str] = []
    radius = fnum(row.get("planet_radius_rearth"))
    depth = fnum(row.get("transit_depth_ppm_or_percent_raw"))
    period = fnum(row.get("period_days"))
    contratio = fnum(row.get("tic_contratio"))
    numcont = fnum(row.get("tic_numcont"))
    odd_sig = fnum(row.get("xml_oddeventransitdepthcomparisonstatistic_significance"))
    secondary_mes = fnum(row.get("xml_weaksecondary_maxmes"))
    if radius >= 8:
        notes.append("large_radius")
    if depth >= 10000:
        notes.append("very_deep")
    elif depth >= 5000:
        notes.append("deep")
    if period < 2:
        notes.append("ultra_short_period")
    if math.isfinite(contratio) and contratio > 0.1:
        notes.append("crowded")
    if math.isfinite(numcont) and numcont >= 100:
        notes.append("many_contaminants")
    if math.isfinite(odd_sig) and odd_sig < 0.01:
        notes.append("odd_even_signal")
    if math.isfinite(secondary_mes) and secondary_mes >= 3:
        notes.append("secondary_signal")
    if not yn(row.get("xml_found", "")):
        notes.append("no_xml")
    if not yn(row.get("dv_has_tce", "")):
        notes.append("no_dv_tce")
    return notes


def audit_row(rank: int, row: dict[str, str], score_column: str) -> dict[str, object]:
    notes = risk_notes(row)
    needs_manual = bool({"large_radius", "very_deep", "deep", "crowded", "many_contaminants", "odd_even_signal", "secondary_signal"} & set(notes))
    return {
        "rank": rank,
        "tic_id": row.get("tic_id", ""),
        "toi": row.get("toi", ""),
        "tfopwg_disp": row.get("tfopwg_disp", ""),
        "strict_planet_class": row.get("strict_planet_class", ""),
        "triage_score": row.get(score_column, row.get("triage_score", "")),
        "score_column": score_column,
        "calibrated_planet_probability": row.get("calibrated_planet_probability", ""),
        "rv_utility": row.get("rv_utility", ""),
        "expected_k_ms": row.get("expected_k_ms", ""),
        "period_days": row.get("period_days", ""),
        "planet_radius_rearth": row.get("planet_radius_rearth", ""),
        "stellar_mass_msun": row.get("stellar_mass_msun", ""),
        "stellar_radius_rsun": row.get("stellar_radius_rsun", ""),
        "tmag": row.get("tmag", ""),
        "transit_depth_ppm_or_percent_raw": row.get("transit_depth_ppm_or_percent_raw", ""),
        "dv_has_tce": row.get("dv_has_tce", ""),
        "xml_found": row.get("xml_found", ""),
        "binary_like_risk": row.get("binary_like_risk", ""),
        "risk_note_count": len(notes),
        "needs_manual_review": int(needs_manual),
        "risk_notes": ";".join(notes) if notes else "none",
    }


def write_markdown(path: pathlib.Path, rows: list[dict[str, object]], summary: dict[str, object], ranking: str, score_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Top-50 Candidate Audit\n\n")
        f.write(f"Input: `{ranking}`\n\n")
        f.write(f"Scope: top {summary['n']} unresolved `PC`/`APC` rows ranked by `{score_column}`.\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Rows audited: {summary['n']}\n")
        f.write(f"- Planet candidates: {summary['class_counts'].get('planet_candidate', 0)}\n")
        f.write(f"- Ambiguous planet candidates: {summary['class_counts'].get('ambiguous_planet_candidate', 0)}\n")
        f.write(f"- With DV TCE: {summary['dv_has_tce']}\n")
        f.write(f"- With DVR XML: {summary['xml_found']}\n")
        f.write(f"- Flagged for manual review: {summary['needs_manual_review']}\n")
        f.write(f"- Median triage score: {summary['median_triage_score']:.3f}\n")
        f.write(f"- Median calibrated planet probability: {summary['median_calibrated_planet_probability']:.3f}\n")
        f.write(f"- Median expected K: {summary['median_expected_k_ms']:.2f} m/s\n\n")
        f.write("Most common risk notes:\n\n")
        for note, count in summary["risk_note_counts"].most_common():  # type: ignore[index]
            f.write(f"- `{note}`: {count}\n")
        f.write("\n## Top 15 Rows\n\n")
        f.write("| Rank | TOI | TIC | Class | Score | p(planet) | K m/s | Radius | Tmag | Risk notes |\n")
        f.write("|---:|---:|---:|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows[:15]:
            f.write(
                f"| {row['rank']} | {row['toi']} | {row['tic_id']} | {row['strict_planet_class']} | "
                f"{float(row['triage_score']):.2f} | {float(row['calibrated_planet_probability']):.3f} | "
                f"{float(row['expected_k_ms']):.1f} | {float(row['planet_radius_rearth']):.1f} | "
                f"{float(row['tmag']):.2f} | `{row['risk_notes']}` |\n"
            )
        f.write("\n## Interpretation\n\n")
        if score_column == "tuned_triage_score":
            f.write("The tuned ranking is selected on strict-label validation metrics, not on unresolved-candidate aesthetics. ")
            f.write("It improves the statistical policy comparison while still surfacing some large-radius or secondary-like unresolved candidates. ")
            f.write("The paper should report this as a validation-tuned policy and compare it with the cleaner hand-penalized queue.\n")
        elif score_column == "revised_triage_score":
            f.write("The revised ranking is less dominated by high-amplitude binary-like systems than the original score. ")
            f.write("It still contains rows with contamination or short-period risk, so it remains a manual-vetting queue rather than an automatic observing list. ")
            f.write("The paper should compare this policy against the original RV-weighted score and inspect representative candidates as case studies.\n")
        else:
            f.write("The top-ranked unresolved list is dominated by high expected RV amplitude systems. ")
            f.write("Many are also large-radius, short-period, deep, or crowded cases, so the table is better treated as a manual-vetting queue than as a clean follow-up recommendation list. ")
            f.write("The paper should use these rows as case studies to show how calibrated probability and RV utility interact, and where binary-like risk still needs stronger modeling.\n")


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", default="reports/candidate_triage_ranking.csv")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--out-csv", default="reports/top50_candidate_audit.csv")
    parser.add_argument("--out-json", default="reports/top50_candidate_audit.json")
    parser.add_argument("--out-md", default="reports/top50_candidate_audit.md")
    parser.add_argument("--score-column", default="triage_score")
    args = parser.parse_args()

    source_rows = read_csv(pathlib.Path(args.ranking))
    audited = [audit_row(i + 1, row, args.score_column) for i, row in enumerate(source_rows[: args.top])]
    write_rows(pathlib.Path(args.out_csv), audited)

    class_counts = Counter(str(row["strict_planet_class"]) for row in audited)
    risk_counts = Counter()
    for row in audited:
        for note in str(row["risk_notes"]).split(";"):
            if note and note != "none":
                risk_counts[note] += 1
    summary: dict[str, object] = {
        "n": len(audited),
        "class_counts": dict(class_counts),
        "dv_has_tce": sum(1 for row in audited if yn(str(row["dv_has_tce"]))),
        "xml_found": sum(1 for row in audited if yn(str(row["xml_found"]))),
        "needs_manual_review": sum(int(row["needs_manual_review"]) for row in audited),
        "risk_note_counts": risk_counts,
        "median_triage_score": median([fnum(row["triage_score"]) for row in audited]),
        "median_calibrated_planet_probability": median([fnum(row["calibrated_planet_probability"]) for row in audited]),
        "median_expected_k_ms": median([fnum(row["expected_k_ms"]) for row in audited]),
    }
    json_summary = dict(summary)
    json_summary["risk_note_counts"] = dict(risk_counts)
    pathlib.Path(args.out_json).write_text(json.dumps(json_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(pathlib.Path(args.out_md), audited, summary, args.ranking, args.score_column)
    print(json.dumps(json_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
