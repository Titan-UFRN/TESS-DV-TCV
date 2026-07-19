#!/usr/bin/env python3
"""Create validation plots from CSV artifacts without heavy plotting dependencies."""

from __future__ import annotations

import argparse
import csv
import html
import pathlib


MODEL_LABELS = {
    "archive_tic": "Archive/TIC",
    "archive_tic_dv_header": "Archive/TIC + DV",
    "archive_tic_xml": "Archive/TIC + XML",
    "all_features": "All features",
}

COLORS = {
    "archive_tic": "#6b7280",
    "archive_tic_dv_header": "#2563eb",
    "archive_tic_xml": "#059669",
    "all_features": "#dc2626",
}


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def text(x: float, y: float, value: str, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#111827">'
        f"{html.escape(value)}</text>"
    )


def rotated_text(x: float, y: float, value: str, angle: int = -90, size: int = 12) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica,Arial,sans-serif" '
        f'font-size="{size}" text-anchor="middle" fill="#111827" '
        f'transform="rotate({angle} {x:.1f} {y:.1f})">{html.escape(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = "#111827", width: float = 1.0, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def polyline(points: list[tuple[float, float]], color: str, width: float = 2.2) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def circle(x: float, y: float, r: float, color: str, opacity: float = 1.0) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" opacity="{opacity:.2f}"/>'


def save_svg(path: pathlib.Path, body: list[str], width: int = 920, height: int = 620) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        *body,
        "</svg>",
        "",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def reliability_plot(rows: list[dict[str, str]], out: pathlib.Path) -> None:
    width, height = 920, 620
    left, right, top, bottom = 90, 250, 64, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(v: float) -> float:
        return left + v * plot_w

    def sy(v: float) -> float:
        return top + (1 - v) * plot_h

    body: list[str] = []
    body.append(text(width / 2, 32, "Reliability Curves", 18, "middle", "700"))
    body.append(line(left, top, left, top + plot_h, "#374151", 1.2))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, "#374151", 1.2))
    body.append(line(sx(0), sy(0), sx(1), sy(1), "#9ca3af", 1.5, "6 5"))
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        body.append(line(sx(tick), top + plot_h, sx(tick), top + plot_h + 5, "#374151", 1))
        body.append(text(sx(tick), top + plot_h + 24, f"{tick:.2f}", 11, "middle"))
        body.append(line(left - 5, sy(tick), left, sy(tick), "#374151", 1))
        body.append(text(left - 12, sy(tick) + 4, f"{tick:.2f}", 11, "end"))
        if tick not in {0, 1.0}:
            body.append(line(left, sy(tick), left + plot_w, sy(tick), "#e5e7eb", 0.8))
            body.append(line(sx(tick), top, sx(tick), top + plot_h, "#e5e7eb", 0.8))
    body.append(text(left + plot_w / 2, height - 28, "Mean calibrated probability", 13, "middle"))
    body.append(rotated_text(25, top + plot_h / 2, "Empirical positive rate", -90, 13))

    by_model: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    for model in MODEL_LABELS:
        pts = []
        for row in sorted(by_model.get(model, []), key=lambda r: int(r["bin"])):
            mp = fnum(row.get("mean_probability", ""))
            ep = fnum(row.get("empirical_positive_rate", ""))
            n = int(float(row.get("n", "0") or 0))
            if mp is None or ep is None or n <= 0:
                continue
            pts.append((sx(mp), sy(ep)))
        if not pts:
            continue
        color = COLORS[model]
        body.append(polyline(pts, color))
        for x, y in pts:
            body.append(circle(x, y, 4.2, color, 0.95))

    lx, ly = width - right + 30, top + 18
    for i, model in enumerate(MODEL_LABELS):
        y = ly + i * 28
        body.append(line(lx, y - 4, lx + 28, y - 4, COLORS[model], 3))
        body.append(text(lx + 38, y, MODEL_LABELS[model], 12))
    body.append(text(lx, ly + 130, "Dashed line = perfect calibration", 11))
    save_svg(out, body, width, height)


def subgroup_plot(rows: list[dict[str, str]], out: pathlib.Path) -> None:
    selected = [r for r in rows if r.get("model") == "all_features"]
    selected = sorted(selected, key=lambda r: r.get("subgroup", ""))
    width, height = 920, 620
    left, right, top, bottom = 225, 95, 62, 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_ece = max([fnum(r.get("ece", "")) or 0 for r in selected] + [0.15])

    def sx(v: float) -> float:
        return left + (v / max_ece) * plot_w

    body: list[str] = []
    body.append(text(width / 2, 32, "Subgroup Calibration: All Features", 18, "middle", "700"))
    body.append(line(left, top, left, top + plot_h, "#374151", 1.2))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, "#374151", 1.2))
    for tick in [0, max_ece / 4, max_ece / 2, max_ece * 3 / 4, max_ece]:
        body.append(line(sx(tick), top, sx(tick), top + plot_h, "#e5e7eb", 0.8))
        body.append(text(sx(tick), top + plot_h + 24, f"{tick:.2f}", 11, "middle"))
    body.append(text(left + plot_w / 2, height - 25, "Expected calibration error", 13, "middle"))

    n = max(1, len(selected))
    step = plot_h / n
    bar_h = min(28, step * 0.58)
    for i, row in enumerate(selected):
        y = top + i * step + step * 0.2
        subgroup = row.get("subgroup", "")
        ece = fnum(row.get("ece", "")) or 0.0
        count = int(float(row.get("n", "0") or 0))
        body.append(text(left - 12, y + bar_h * 0.7, subgroup, 11, "end"))
        body.append(f'<rect x="{left:.1f}" y="{y:.1f}" width="{max(1, sx(ece)-left):.1f}" height="{bar_h:.1f}" fill="#dc2626" opacity="0.82"/>')
        body.append(text(sx(ece) + 6, y + bar_h * 0.7, f"{ece:.3f} (n={count})", 10))
    save_svg(out, body, width, height)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reliability", default="reports/validation/reliability_curve.csv")
    parser.add_argument("--subgroup", default="reports/validation/subgroup_calibration.csv")
    parser.add_argument("--outdir", default="reports/validation")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    reliability_plot(read_csv(pathlib.Path(args.reliability)), outdir / "reliability_curve.svg")
    subgroup_plot(read_csv(pathlib.Path(args.subgroup)), outdir / "subgroup_calibration.svg")
    print(f"wrote {outdir / 'reliability_curve.svg'}")
    print(f"wrote {outdir / 'subgroup_calibration.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
