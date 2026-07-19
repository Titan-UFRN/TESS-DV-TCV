#!/usr/bin/env python3
"""Create a revised candidate ranking with explicit vetting penalties."""

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
    elif radius >= 6:
        notes.append("moderately_large_radius")
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


PENALTY_BY_NOTE = {
    "large_radius": 0.20,
    "moderately_large_radius": 0.55,
    "very_deep": 0.35,
    "deep": 0.60,
    "ultra_short_period": 0.75,
    "crowded": 0.65,
    "many_contaminants": 0.70,
    "odd_even_signal": 0.45,
    "secondary_signal": 0.45,
    "no_xml": 0.70,
    "no_dv_tce": 0.70,
}


def vetting_penalty(notes: list[str]) -> float:
    penalty = 1.0
    for note in notes:
        penalty *= PENALTY_BY_NOTE.get(note, 1.0)
    return max(0.02, min(1.0, penalty))


def needs_manual_review(notes: list[str]) -> bool:
    hard_notes = {
        "large_radius",
        "very_deep",
        "deep",
        "crowded",
        "many_contaminants",
        "odd_even_signal",
        "secondary_signal",
    }
    return bool(hard_notes & set(notes))


def revise(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    revised: list[dict[str, object]] = []
    for row in rows:
        notes = risk_notes(row)
        penalty = vetting_penalty(notes)
        p_planet = fnum(row.get("calibrated_planet_probability"))
        rv_utility = fnum(row.get("rv_utility"))
        original = fnum(row.get("triage_score"))
        if not math.isfinite(p_planet):
            p_planet = 0.0
        if not math.isfinite(rv_utility):
            rv_utility = 0.0
        revised_score = p_planet * rv_utility * penalty
        out: dict[str, object] = dict(row)
        out["original_triage_score"] = original if math.isfinite(original) else ""
        out["vetting_penalty"] = penalty
        out["vetting_risk_notes"] = ";".join(notes) if notes else "none"
        out["vetting_risk_note_count"] = len(notes)
        out["needs_manual_review"] = int(needs_manual_review(notes))
        out["revised_triage_score"] = revised_score
        revised.append(out)
    revised.sort(key=lambda row: fnum(row.get("revised_triage_score")), reverse=True)
    return revised


def summarize(rows: list[dict[str, object]], top: int) -> dict[str, object]:
    selected = rows[:top]
    risk_counts = Counter()
    class_counts = Counter(str(row.get("strict_planet_class", "")) for row in selected)
    for row in selected:
        for note in str(row.get("vetting_risk_notes", "")).split(";"):
            if note and note != "none":
                risk_counts[note] += 1
    return {
        "n": len(selected),
        "class_counts": dict(class_counts),
        "needs_manual_review": sum(int(row.get("needs_manual_review", 0)) for row in selected),
        "risk_note_counts": dict(risk_counts),
        "median_revised_triage_score": median([fnum(row.get("revised_triage_score")) for row in selected]),
        "median_calibrated_planet_probability": median([fnum(row.get("calibrated_planet_probability")) for row in selected]),
        "median_expected_k_ms": median([fnum(row.get("expected_k_ms")) for row in selected]),
        "median_vetting_penalty": median([fnum(row.get("vetting_penalty")) for row in selected]),
    }


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def write_comparison(path: pathlib.Path, before: dict[str, object], after: dict[str, object], top: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    before_risks = before.get("risk_note_counts", {})
    after_risks = after.get("risk_note_counts", {})
    all_notes = sorted(set(before_risks) | set(after_risks))  # type: ignore[arg-type]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Revised Triage Scoring Comparison\n\n")
        f.write(f"Scope: top {top} unresolved candidates before and after explicit vetting penalties.\n\n")
        f.write("## Summary\n\n")
        f.write("| Metric | Original | Revised |\n")
        f.write("|---|---:|---:|\n")
        f.write(f"| Manual-review flags | {before['needs_manual_review']} | {after['needs_manual_review']} |\n")
        f.write(f"| Median calibrated p(planet) | {before['median_calibrated_planet_probability']:.3f} | {after['median_calibrated_planet_probability']:.3f} |\n")
        f.write(f"| Median expected K (m/s) | {before['median_expected_k_ms']:.2f} | {after['median_expected_k_ms']:.2f} |\n")
        f.write(f"| Median score | {before['median_triage_score']:.3f} | {after['median_revised_triage_score']:.3f} |\n")
        f.write(f"| Median vetting penalty | 1.000 | {after['median_vetting_penalty']:.3f} |\n\n")
        f.write("## Risk Counts\n\n")
        f.write("| Risk note | Original top 50 | Revised top 50 | Delta |\n")
        f.write("|---|---:|---:|---:|\n")
        for note in all_notes:
            b = int(before_risks.get(note, 0))  # type: ignore[union-attr]
            a = int(after_risks.get(note, 0))  # type: ignore[union-attr]
            f.write(f"| `{note}` | {b} | {a} | {a - b} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("The revised score intentionally trades high expected RV amplitude for stronger photometric and scene-vetting hygiene. ")
        f.write("It should be treated as a second triage policy rather than as a final classifier: the remaining top-ranked rows still need manual review, but the ranked list is less dominated by obvious large-radius, deep, or crowded cases.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", default="reports/candidate_triage_ranking.csv")
    parser.add_argument("--before-summary", default="reports/top50_candidate_audit.json")
    parser.add_argument("--out-ranking", default="reports/revised_candidate_triage_ranking.csv")
    parser.add_argument("--out-summary", default="reports/revised_triage_comparison.json")
    parser.add_argument("--out-md", default="reports/revised_triage_comparison.md")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    rows = read_csv(pathlib.Path(args.ranking))
    revised = revise(rows)
    write_rows(pathlib.Path(args.out_ranking), revised)

    before = json.loads(pathlib.Path(args.before_summary).read_text(encoding="utf-8"))
    after = summarize(revised, args.top)
    comparison = {
        "top": args.top,
        "penalty_by_note": PENALTY_BY_NOTE,
        "before": before,
        "after": after,
    }
    pathlib.Path(args.out_summary).write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_comparison(pathlib.Path(args.out_md), before, after, args.top)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
