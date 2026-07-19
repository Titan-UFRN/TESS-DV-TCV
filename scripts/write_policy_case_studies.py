#!/usr/bin/env python3
"""Write side-by-side candidate case summaries for triage policies."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib

from revise_candidate_scoring import risk_notes


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
        return out if math.isfinite(out) else math.nan
    except (TypeError, ValueError):
        return math.nan


def row_key(row: dict[str, str]) -> str:
    return f"{row.get('tic_id', '')}:{row.get('toi', '')}"


def format_row(rank: int, row: dict[str, str], score_col: str) -> str:
    notes = risk_notes(row)
    note_text = ";".join(notes) if notes else "none"
    score = fnum(row.get(score_col))
    p_planet = fnum(row.get("calibrated_planet_probability"))
    k_ms = fnum(row.get("expected_k_ms"))
    radius = fnum(row.get("planet_radius_rearth"))
    tmag = fnum(row.get("tmag"))
    return (
        f"| {rank} | {row.get('toi', '')} | {row.get('tic_id', '')} | "
        f"{row.get('strict_planet_class', '')} | {score:.2f} | {p_planet:.3f} | "
        f"{k_ms:.1f} | {radius:.1f} | {tmag:.2f} | `{note_text}` |"
    )


def write_policy_section(f, title: str, path: pathlib.Path, score_col: str, top: int) -> set[str]:
    rows = read_csv(path)[:top]
    f.write(f"## {title}\n\n")
    f.write(f"Source: `{path}` ranked by `{score_col}`.\n\n")
    f.write("| Rank | TOI | TIC | Class | Score | p(planet) | K m/s | Radius | Tmag | Risk notes |\n")
    f.write("|---:|---:|---:|---|---:|---:|---:|---:|---:|---|\n")
    for i, row in enumerate(rows, start=1):
        f.write(format_row(i, row, score_col) + "\n")
    f.write("\n")
    return {row_key(row) for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", default="reports/candidate_triage_ranking.csv")
    parser.add_argument("--hand", default="reports/revised_candidate_triage_ranking.csv")
    parser.add_argument("--tuned", default="reports/tuned_candidate_triage_ranking.csv")
    parser.add_argument("--out", default="reports/policy_case_studies.md")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# Policy Case Studies\n\n")
        f.write("This file compares the highest-ranked unresolved candidates under three scoring policies.\n\n")
        original = write_policy_section(f, "Original RV-Weighted Score", pathlib.Path(args.original), "triage_score", args.top)
        hand = write_policy_section(f, "Hand-Penalized Score", pathlib.Path(args.hand), "revised_triage_score", args.top)
        tuned = write_policy_section(f, "Validation-Tuned Penalty Score", pathlib.Path(args.tuned), "tuned_triage_score", args.top)
        f.write("## Overlap\n\n")
        f.write(f"- Original vs hand top {args.top}: {len(original & hand)} shared rows\n")
        f.write(f"- Original vs tuned top {args.top}: {len(original & tuned)} shared rows\n")
        f.write(f"- Hand vs tuned top {args.top}: {len(hand & tuned)} shared rows\n\n")
        f.write("## Interpretation\n\n")
        f.write("The original policy surfaces high-amplitude but often binary-like systems. ")
        f.write("The hand-penalized policy is cleaner for human follow-up triage. ")
        f.write("The tuned policy is selected on strict-label validation metrics and should be reported as the statistically tuned policy, even when the hand policy is cleaner in the unresolved top list.\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
