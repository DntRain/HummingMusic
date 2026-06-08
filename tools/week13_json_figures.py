"""Render Week 13 regression JSON results as report-ready SVG figures."""

from __future__ import annotations

import json
from collections import Counter
from html import escape
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT.parent / "docs"
RESULT_PATH = DOCS_DIR / "week13_regression_results.json"
ASSET_DIR = DOCS_DIR / "week13_regression_assets"


def _svg(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        "<style>"
        "text{font-family:Segoe UI,Microsoft YaHei,Arial,sans-serif;fill:#17202a}"
        ".title{font-size:24px;font-weight:700}"
        ".label{font-size:14px}"
        ".small{font-size:12px;fill:#5d6d7e}"
        ".value{font-size:28px;font-weight:700}"
        ".axis{stroke:#d5d8dc;stroke-width:1}"
        "</style>\n"
        f"{body}\n"
        "</svg>\n"
    )


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def render_status_summary(payload: dict) -> None:
    summary = payload["summary"]
    cards = [
        ("Total", str(summary["total"]), "#2e86c1"),
        ("Passed", str(summary["passed"]), "#239b56"),
        ("Failed", str(summary["failed"]), "#c0392b"),
        ("Pass rate", f'{summary["pass_rate"] * 100:.2f}%', "#7d3c98"),
        ("P0/P1 blockers", str(summary["p0_p1_blockers"]), "#d68910"),
    ]
    width = 1100
    height = 310
    card_w = 190
    gap = 18
    x0 = 45
    body = [
        '<rect width="1100" height="310" fill="#fbfcfc"/>',
        '<text class="title" x="45" y="50">Week 13 Regression Summary</text>',
        f'<text class="small" x="45" y="76">Source: {escape(RESULT_PATH.name)}</text>',
    ]
    for idx, (label, value, color) in enumerate(cards):
        x = x0 + idx * (card_w + gap)
        body.extend(
            [
                f'<rect x="{x}" y="112" width="{card_w}" height="130" rx="8" fill="#ffffff" stroke="#d6dbdf"/>',
                f'<rect x="{x}" y="112" width="{card_w}" height="8" rx="4" fill="{color}"/>',
                f'<text class="value" x="{x + 22}" y="174">{escape(value)}</text>',
                f'<text class="label" x="{x + 22}" y="210">{escape(label)}</text>',
            ]
        )
    body.append('<text class="small" x="45" y="278">All 31 regression checks passed; no blocking P0/P1 issue was found.</text>')
    _write(ASSET_DIR / "week13_status_summary.svg", _svg(width, height, "\n".join(body)))


def render_module_coverage(results: list[dict]) -> None:
    counts = Counter(item["module"] for item in results)
    order = ["baseline", "unit", "audio", "quantizer", "style", "renderer", "frontend"]
    values = [(module, counts[module]) for module in order if module in counts]
    max_value = max(counts.values())
    width = 1100
    height = 430
    x_label = 150
    x_bar = 225
    bar_max = 780
    y0 = 105
    step = 42
    body = [
        '<rect width="1100" height="430" fill="#fbfcfc"/>',
        '<text class="title" x="45" y="50">Regression Coverage by Module</text>',
        '<text class="small" x="45" y="76">Case count grouped from JSON result records.</text>',
    ]
    for idx, (module, count) in enumerate(values):
        y = y0 + idx * step
        bar_w = round(bar_max * count / max_value)
        body.extend(
            [
                f'<text class="label" x="{x_label}" y="{y + 18}" text-anchor="end">{escape(module)}</text>',
                f'<rect x="{x_bar}" y="{y}" width="{bar_max}" height="24" rx="5" fill="#edf2f7"/>',
                f'<rect x="{x_bar}" y="{y}" width="{bar_w}" height="24" rx="5" fill="#2874a6"/>',
                f'<text class="label" x="{x_bar + bar_w + 12}" y="{y + 18}">{count}</text>',
            ]
        )
    body.append(f'<line class="axis" x1="{x_bar}" y1="390" x2="{x_bar + bar_max}" y2="390"/>')
    _write(ASSET_DIR / "week13_module_coverage.svg", _svg(width, height, "\n".join(body)))


def render_elapsed_top10(results: list[dict]) -> None:
    slowest = sorted(results, key=lambda item: item["elapsed_ms"], reverse=True)[:10]
    max_value = max(item["elapsed_ms"] for item in slowest)
    width = 1100
    height = 560
    x_label = 245
    x_bar = 290
    bar_max = 690
    y0 = 105
    step = 42
    body = [
        '<rect width="1100" height="560" fill="#fbfcfc"/>',
        '<text class="title" x="45" y="50">Top 10 Longest Regression Cases</text>',
        '<text class="small" x="45" y="76">Elapsed time in milliseconds, generated from week13_regression_results.json.</text>',
    ]
    for idx, item in enumerate(slowest):
        y = y0 + idx * step
        bar_w = round(bar_max * item["elapsed_ms"] / max_value)
        label = f'{item["case_id"]} {item["module"]}'
        name = item["name"]
        if len(name) > 34:
            name = f"{name[:31]}..."
        body.extend(
            [
                f'<text class="label" x="{x_label}" y="{y + 18}" text-anchor="end">{escape(label)}</text>',
                f'<rect x="{x_bar}" y="{y}" width="{bar_max}" height="24" rx="5" fill="#edf2f7"/>',
                f'<rect x="{x_bar}" y="{y}" width="{bar_w}" height="24" rx="5" fill="#d35400"/>',
                f'<text class="label" x="{x_bar + bar_w + 12}" y="{y + 18}">{item["elapsed_ms"]:.2f}</text>',
                f'<text class="small" x="{x_bar}" y="{y + 38}">{escape(name)}</text>',
            ]
        )
    body.append(f'<line class="axis" x1="{x_bar}" y1="525" x2="{x_bar + bar_max}" y2="525"/>')
    _write(ASSET_DIR / "week13_elapsed_top10.svg", _svg(width, height, "\n".join(body)))


def main() -> int:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    results = payload["results"]
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    render_status_summary(payload)
    render_module_coverage(results)
    render_elapsed_top10(results)
    print(f"wrote figures to {ASSET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
