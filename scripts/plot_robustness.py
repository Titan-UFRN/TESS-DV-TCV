#!/usr/bin/env python3
"""Plot repeated-split intervals for the principal model comparisons."""

from __future__ import annotations

import argparse
import csv
import html
import pathlib


LABELS = {
    "archive_tic": "Archive/TIC",
    "archive_tic_dv_header": "+ DV header",
    "archive_tic_xml": "+ DVR XML",
    "all_features": "All features",
}
COLORS = {
    "archive_tic": "#6b7280",
    "archive_tic_dv_header": "#2563a6",
    "archive_tic_xml": "#b45309",
    "all_features": "#167a5b",
}


def read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def scale(value: float, lo: float, hi: float, left: float, right: float) -> float:
    return left + (value - lo) / (hi - lo) * (right - left)


def panel(
    rows: list[dict[str, str]],
    x0: int,
    width: int,
    title: str,
    field: str,
    lo: float,
    hi: float,
    ticks: list[float],
) -> list[str]:
    left = x0 + 122
    right = x0 + width - 24
    top = 74
    bottom = 342
    svg = [f'<text x="{x0 + width / 2:.1f}" y="30" text-anchor="middle" class="title">{html.escape(title)}</text>']
    for tick in ticks:
        x = scale(tick, lo, hi, left, right)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>')
        svg.append(f'<text x="{x:.1f}" y="370" text-anchor="middle" class="tick">{tick:.2f}</text>')
    for i, row in enumerate(rows):
        model = row["model"]
        y = 105 + i * 68
        mean = float(row[f"{field}_mean"])
        low = float(row[f"{field}_q025"])
        high = float(row[f"{field}_q975"])
        x_mean = scale(mean, lo, hi, left, right)
        x_low = scale(low, lo, hi, left, right)
        x_high = scale(high, lo, hi, left, right)
        color = COLORS[model]
        svg.append(f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" class="label">{html.escape(LABELS[model])}</text>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y}" x2="{x_high:.1f}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        svg.append(f'<line x1="{x_low:.1f}" y1="{y - 7}" x2="{x_low:.1f}" y2="{y + 7}" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<line x1="{x_high:.1f}" y1="{y - 7}" x2="{x_high:.1f}" y2="{y + 7}" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<circle cx="{x_mean:.1f}" cy="{y}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    return svg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="reports/robustness/repeated_split_summary.csv")
    parser.add_argument("--out", default="reports/robustness/repeated_split_intervals.svg")
    args = parser.parse_args()

    rows = read_rows(pathlib.Path(args.summary))
    order = list(LABELS)
    by_model = {row["model"]: row for row in rows}
    rows = [by_model[model] for model in order]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="410" viewBox="0 0 1000 410">',
        '<style>',
        'text { font-family: Helvetica, Arial, sans-serif; fill: #17202a; }',
        '.title { font-size: 20px; font-weight: 700; }',
        '.label { font-size: 15px; }',
        '.tick { font-size: 13px; fill: #4b5563; }',
        '.grid { stroke: #d7dce2; stroke-width: 1; }',
        '.axis { stroke: #6b7280; stroke-width: 1.5; }',
        '</style>',
        '<rect width="1000" height="410" fill="#ffffff"/>',
    ]
    svg.extend(panel(rows, 0, 500, "Brier score (lower is better)", "brier", 0.12, 0.23, [0.12, 0.15, 0.18, 0.21]))
    svg.extend(panel(rows, 500, 500, "Precision at 100 (higher is better)", "precision_at_100", 0.60, 0.98, [0.60, 0.70, 0.80, 0.90]))
    svg.append('<text x="500" y="402" text-anchor="middle" class="tick">Points are means across 20 TIC-grouped splits; bars are empirical 2.5th--97.5th percentile split intervals.</text>')
    svg.append('</svg>')
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
