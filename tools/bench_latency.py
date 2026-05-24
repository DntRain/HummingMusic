"""
端到端 pipeline 延迟基线测量。

用法：
    python -m tools.bench_latency \
        --audios example_1.m4a example_2.m4a data/demo/example.wav \
        --styles pop jazz classical folk \
        --runs 3 \
        --out reports/week12/baseline.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from src.interfaces import extract_pitch, quantize_humming, render_audio, transfer_style

ROOT = Path(__file__).resolve().parent.parent
STAGES = ["extract_pitch", "quantize_humming", "transfer_style", "render_audio"]


def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def run_once(audio: str, style: str) -> dict[str, float]:
    record: dict[str, float] = {}
    pitch, record["extract_pitch"] = time_call(extract_pitch, audio)
    midi, record["quantize_humming"] = time_call(quantize_humming, pitch)
    styled, record["transfer_style"] = time_call(transfer_style, midi, style)
    _, record["render_audio"] = time_call(render_audio, styled)
    record["total"] = sum(record[s] for s in STAGES)
    return record


def summarize(samples: list[dict[str, float]]) -> dict:
    keys = STAGES + ["total"]
    out = {}
    for k in keys:
        vals = [s[k] for s in samples]
        out[k] = {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audios", nargs="+", required=True)
    ap.add_argument("--styles", nargs="+",
                    default=["pop", "jazz", "classical", "folk"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1,
                    help="预热轮数（不计入统计），用于消除冷启动/模型加载")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    results = {"meta": {
        "audios": args.audios, "styles": args.styles,
        "runs": args.runs, "warmup": args.warmup,
    }, "per_case": [], "per_stage_overall": {}, "per_audio_overall": {}}

    pooled: list[dict[str, float]] = []

    for audio in args.audios:
        if not Path(audio).exists():
            print(f"[skip] {audio} not found")
            continue
        for style in args.styles:
            samples: list[dict[str, float]] = []
            for w in range(args.warmup):
                print(f"[warmup {w + 1}/{args.warmup}] {audio} | {style}")
                run_once(audio, style)
            for r in range(args.runs):
                rec = run_once(audio, style)
                samples.append(rec)
                print(f"[run {r + 1}/{args.runs}] {audio} | {style} | "
                      f"total={rec['total']:.2f}s")
            case = {
                "audio": audio,
                "style": style,
                "samples": samples,
                "summary": summarize(samples),
            }
            results["per_case"].append(case)
            pooled.extend(samples)

    if pooled:
        results["per_stage_overall"] = summarize(pooled)
        for audio in args.audios:
            sub = [s for c in results["per_case"] if c["audio"] == audio
                   for s in c["samples"]]
            if sub:
                results["per_audio_overall"][audio] = summarize(sub)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print("\n========== OVERALL ==========")
    if pooled:
        s = results["per_stage_overall"]
        for k in STAGES + ["total"]:
            print(f"  {k:<18s} mean={s[k]['mean']:.3f}s  "
                  f"median={s[k]['median']:.3f}s  "
                  f"max={s[k]['max']:.3f}s")
        target = 10.0
        passed = s["total"]["mean"] <= target
        print(f"\n  目标 ≤ {target:.1f}s，当前均值 {s['total']['mean']:.2f}s → "
              f"{'PASS' if passed else 'FAIL'}")
    print(f"\nresult: {out_path}")


if __name__ == "__main__":
    main()
