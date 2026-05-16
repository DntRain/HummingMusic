"""
Week 10 profiling for Zhong Xiang's integration task.

Runs a deterministic synthetic humming pipeline:
pitch frames -> quantizer -> 4 style transfers -> WAV rendering.
It compares the legacy music21 key-analysis path with the optimized
pitch-class key estimator used by default.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.interfaces import quantize_humming, render_audio, transfer_style
from src import style_transfer as style_module


STYLES = ["pop", "jazz", "classical", "folk"]


def make_pitch_data(duration: float = 8.0, step: float = 0.01) -> dict:
    """Build a repeatable C-major humming-like F0 sequence."""
    times = np.arange(0.0, duration, step)
    pattern_hz = np.array([261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25])
    note_span = max(int(0.5 / step), 1)
    frequency = np.empty_like(times)
    for i in range(len(times)):
        frequency[i] = pattern_hz[(i // note_span) % len(pattern_hz)]

    # Insert two longer pauses so the quantizer exercises note splitting.
    confidence = np.full(len(times), 0.95, dtype=np.float32)
    for start_s, end_s in [(2.0, 2.28), (5.0, 5.32)]:
        mask = (times >= start_s) & (times < end_s)
        confidence[mask] = 0.2

    return {
        "time": times,
        "frequency": frequency,
        "confidence": confidence,
        "bpm": 120.0,
    }


def timed(label: str, func):
    started = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - started
    return label, elapsed, result


def run_once(fast_chord_inference: bool, include_render: bool) -> dict:
    style_module._config["style_transfer"]["fast_chord_inference"] = fast_chord_inference
    result: dict = {
        "fast_chord_inference": fast_chord_inference,
        "include_render": include_render,
        "stages": {},
        "styles": {},
    }

    _, quantize_s, melody_midi = timed(
        "quantize", lambda: quantize_humming(make_pitch_data())
    )
    result["stages"]["quantize_s"] = quantize_s
    result["note_count"] = sum(len(inst.notes) for inst in melody_midi.instruments)

    styled_midis = {}
    transfer_total = 0.0
    for style in STYLES:
        _, elapsed, styled = timed(
            f"transfer_{style}", lambda s=style: transfer_style(melody_midi, s)
        )
        styled_midis[style] = styled
        transfer_total += elapsed
        result["styles"][style] = {
            "transfer_s": elapsed,
            "track_count": len(styled.instruments),
            "note_count": sum(len(inst.notes) for inst in styled.instruments),
        }
    result["stages"]["transfer_all_s"] = transfer_total

    render_total = 0.0
    if include_render:
        for style, styled in styled_midis.items():
            _, elapsed, wav_path = timed(
                f"render_{style}", lambda midi=styled: render_audio(midi)
            )
            render_total += elapsed
            result["styles"][style]["render_s"] = elapsed
            result["styles"][style]["wav_path"] = wav_path
            result["styles"][style]["wav_kb"] = Path(wav_path).stat().st_size / 1024.0
    result["stages"]["render_all_s"] = render_total
    result["stages"]["total_s"] = quantize_s + transfer_total + render_total
    return result


def summarize(runs: list[dict]) -> dict:
    totals = [r["stages"]["total_s"] for r in runs]
    transfers = [r["stages"]["transfer_all_s"] for r in runs]
    renders = [r["stages"]["render_all_s"] for r in runs]
    return {
        "runs": len(runs),
        "total_avg_s": statistics.mean(totals),
        "total_min_s": min(totals),
        "total_max_s": max(totals),
        "transfer_avg_s": statistics.mean(transfers),
        "render_avg_s": statistics.mean(renders),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default="experiments/week10_performance/profile_results.json")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    include_render = not args.skip_render
    baseline = [run_once(False, include_render) for _ in range(args.runs)]
    optimized = [run_once(True, include_render) for _ in range(args.runs)]

    payload = {
        "scenario": "8s synthetic humming, 4 styles, fallback local models absent",
        "baseline": {
            "name": "music21 key analysis",
            "summary": summarize(baseline),
            "runs": baseline,
        },
        "optimized": {
            "name": "fast pitch-class key inference",
            "summary": summarize(optimized),
            "runs": optimized,
        },
    }
    base_total = payload["baseline"]["summary"]["total_avg_s"]
    opt_total = payload["optimized"]["summary"]["total_avg_s"]
    payload["improvement"] = {
        "total_saved_s": base_total - opt_total,
        "total_speedup_x": base_total / opt_total if opt_total > 0 else None,
        "meets_10s_target": opt_total <= 10.0,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "baseline_total_avg_s": base_total,
        "optimized_total_avg_s": opt_total,
        "total_speedup_x": payload["improvement"]["total_speedup_x"],
        "meets_10s_target": payload["improvement"]["meets_10s_target"],
        "output": str(out_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
