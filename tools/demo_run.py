#!/usr/bin/env python3
"""
tools/demo_run.py — HummingMusic 最终演示驱动脚本（dc / week14）

演示时直接执行本脚本，它会把系统**从头到尾**跑一遍并逐阶段可见地输出：
    哼唱音频 → 音高提取(torchcrepe) → 容错量化(BiLSTM-CRF/baseline)
            → 四风格迁移(VQ-VAE + 后处理) → 音频渲染(FluidSynth/pretty_midi)
每个风格产出可播放 WAV + 1080p 钢琴卷帘 PNG，便于录屏展示。

────────────────────────────────────────────────────────────
用法
    # 默认：跑仓库根目录的 example_1/2/3.m4a，全部 4 风格
    python tools/demo_run.py

    # 指定输入 + 演示停顿（每阶段按 Enter 继续，方便边讲解边走）
    python tools/demo_run.py --inputs example_1.m4a --pause

    # 指定输出目录 / 只跑部分风格 / 关闭画图
    python tools/demo_run.py -o demo_output --styles pop jazz --no-plot

录制建议（满足 ≥3min / 4 风格 / ≥1080p 要求）
    • 屏幕/录屏分辨率设 1920×1080（OBS：基准+输出分辨率均 1920×1080，60fps）。
    • 终端字号调大、配色高对比；本脚本输出带分隔横幅，录屏清晰。
    • 加 --pause，在每阶段停顿处口播讲解（提取/量化/4 风格各讲一段），
      3 个样本 × 4 风格自然超过 3 分钟；只跑 1 个样本时建议配 --pause。
    • 生成的 PNG 为 1920×1080，可在视频里全屏插入；WAV 可现场播放对比。
────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("FLUID_NO_AUDIO_DRIVERS", "1")  # headless 渲染安全

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STYLES_ALL = ["pop", "jazz", "classical", "folk"]
STYLE_CN = {"pop": "流行 Pop", "jazz": "爵士 Jazz",
            "classical": "古典 Classical", "folk": "民谣 Folk"}

# ── ANSI 配色（录屏高对比）──────────────────────────────
_C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "cyan": "\033[96m", "green": "\033[92m", "yellow": "\033[93m",
    "magenta": "\033[95m", "blue": "\033[94m", "red": "\033[91m",
}


def c(text: str, *styles: str) -> str:
    if os.environ.get("NO_COLOR"):
        return text
    return "".join(_C[s] for s in styles) + text + _C["reset"]


def banner(title: str, color: str = "cyan") -> None:
    line = "═" * 64
    print("\n" + c(line, color, "bold"))
    print(c(f"  {title}", color, "bold"))
    print(c(line, color, "bold"))


def step(msg: str) -> None:
    print(c("  ▶ ", "green", "bold") + msg)


def info(k: str, v: str) -> None:
    print(f"    {c(k, 'dim'):<22} {c(v, 'yellow')}")


def wait(pause: bool, prompt: str = "按 Enter 继续…") -> None:
    if pause:
        try:
            input(c(f"\n  ⏸  {prompt}", "magenta", "bold"))
        except (EOFError, KeyboardInterrupt):
            print()


# ── 1080p 钢琴卷帘 ─────────────────────────────────────
def plot_piano_roll(midi, title: str, out_png: Path) -> bool:
    """把 PrettyMIDI 画成 1920×1080 钢琴卷帘 PNG。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(c(f"    （跳过画图：matplotlib 不可用 {e}）", "dim"))
        return False

    notes = [(n.start, n.end, n.pitch, inst.name)
             for inst in midi.instruments for n in inst.notes]
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)  # → 1920×1080
    if notes:
        names = sorted({n[3] for n in notes})
        cmap = plt.get_cmap("tab10")
        color_of = {nm: cmap(i % 10) for i, nm in enumerate(names)}
        for s, e, p, nm in notes:
            ax.add_patch(plt.Rectangle((s, p - 0.45), max(e - s, 0.02), 0.9,
                                       facecolor=color_of[nm],
                                       edgecolor="black", linewidth=0.4))
        pitches = [n[2] for n in notes]
        ends = [n[1] for n in notes]
        ax.set_xlim(0, max(ends) * 1.02)
        ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_of[nm])
                   for nm in names]
        ax.legend(handles, names, loc="upper right", fontsize=14)
    else:
        ax.text(0.5, 0.5, "（空 MIDI）", ha="center", va="center",
                transform=ax.transAxes, fontsize=24)
    ax.set_xlabel("时间 / 秒  Time (s)", fontsize=16)
    ax.set_ylabel("MIDI 音高  Pitch", fontsize=16)
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_png))
    plt.close(fig)
    return True


# ── 主流程 ─────────────────────────────────────────────
def run_one(audio_path: Path, styles: list[str], out_dir: Path,
            do_plot: bool, pause: bool) -> dict:
    from src.audio_processing import extract_pitch
    from src.quantizer import quantize_humming
    from src.style_postprocess import stylize
    from src.renderer import render_audio
    import numpy as np

    sample_dir = out_dir / audio_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    result = {"sample": audio_path.name, "styles": {}}

    banner(f"样本：{audio_path.name}", "blue")
    info("输出目录", str(sample_dir))
    wait(pause, "开始处理该样本，按 Enter…")

    # 阶段 1：音高提取
    banner("① 音高提取  Pitch Extraction (torchcrepe)", "cyan")
    t0 = time.time()
    pitch = extract_pitch(str(audio_path))
    n = len(pitch["time"])
    valid = int(np.sum(np.isfinite(pitch["frequency"])))
    step(f"提取完成（{time.time() - t0:.1f}s）")
    info("总帧数", str(n))
    info("有效帧", f"{valid}/{n}  ({100 * valid / max(n, 1):.0f}%)")
    info("置信度 max/mean",
         f"{np.nanmax(pitch['confidence']):.2f} / "
         f"{np.nanmean(pitch['confidence']):.2f}")
    info("估计 BPM", f"{pitch['bpm']:.1f}")
    wait(pause, "讲解音高提取后，按 Enter 进入量化…")

    # 阶段 2：容错量化
    banner("② 容错量化  Quantization (BiLSTM-CRF → MIDI)", "cyan")
    t0 = time.time()
    melody = quantize_humming(pitch)
    mn = sum(len(i.notes) for i in melody.instruments)
    step(f"量化完成（{time.time() - t0:.1f}s）")
    info("旋律音符数", str(mn))
    if mn == 0:
        print(c("    ⚠ 该样本未量化出音符（请换一段哼唱）", "red", "bold"))
    melody_mid = sample_dir / "melody.mid"
    melody.write(str(melody_mid))
    info("旋律 MIDI", str(melody_mid))
    if do_plot:
        png = sample_dir / "melody_pianoroll.png"
        if plot_piano_roll(melody, f"{audio_path.stem} — 量化旋律", png):
            info("旋律卷帘图", str(png))
    result["melody_notes"] = mn
    wait(pause, "讲解量化结果后，按 Enter 进入四风格迁移…")

    # 阶段 3：四风格迁移 + 渲染（stylize 多轨：旋律+和弦+bass+鼓，与 app 一致）
    banner("③ 四风格迁移 + 渲染  Style Transfer × 4 + Render", "cyan")
    bpm = float(pitch.get("bpm") or 120.0)
    for st in styles:
        print()
        step(c(f"风格：{STYLE_CN.get(st, st)}", "bold"))
        t0 = time.time()
        styled = stylize(melody, st, tempo=bpm)
        tn = {i.name: len(i.notes) for i in styled.instruments}
        total = sum(tn.values())
        wav_src = render_audio(styled)          # tmp/{uuid}/output.wav
        wav_dst = sample_dir / f"{st}.wav"
        mid_dst = sample_dir / f"{st}.mid"
        shutil.copy(wav_src, wav_dst)
        styled.write(str(mid_dst))
        size_kb = wav_dst.stat().st_size // 1024
        info("总音符 / 轨道", f"{total}  {tn}")
        info("渲染 WAV", f"{wav_dst}  ({size_kb} KB, {time.time() - t0:.1f}s)")
        if do_plot:
            png = sample_dir / f"{st}_pianoroll.png"
            if plot_piano_roll(styled, f"{audio_path.stem} — {STYLE_CN.get(st, st)}",
                               png):
                info("卷帘图", str(png))
        result["styles"][st] = {"notes": total, "wav": str(wav_dst)}
        wait(pause, f"播放 {st}.wav 并讲解后，按 Enter 继续…")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="HummingMusic 演示驱动脚本：从头跑通整个系统")
    ap.add_argument("--inputs", "-i", nargs="+",
                    help="输入哼唱音频（默认 example_1/2/3.m4a）")
    ap.add_argument("--styles", "-s", nargs="+", default=STYLES_ALL,
                    choices=STYLES_ALL, help="要演示的风格（默认 4 种全跑）")
    ap.add_argument("--out", "-o", default="demo_output",
                    help="输出目录（默认 demo_output/）")
    ap.add_argument("--pause", action="store_true",
                    help="每阶段停顿等 Enter（边讲边走，推荐演示用）")
    ap.add_argument("--no-plot", action="store_true",
                    help="不生成钢琴卷帘 PNG")
    args = ap.parse_args()

    # 静音三方库 INFO 日志，保持演示输出干净
    import logging
    logging.getLogger().setLevel(logging.WARNING)

    if args.inputs:
        inputs = [Path(p) for p in args.inputs]
    else:
        inputs = [ROOT / f"example_{i}.m4a" for i in (1, 2, 3)]
    inputs = [p for p in inputs if p.exists()]
    if not inputs:
        print(c("✗ 找不到输入音频。请用 --inputs 指定，或把 example_*.m4a "
                "放到仓库根目录。", "red", "bold"))
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    banner("HummingMusic — 最终系统演示", "magenta")
    info("输入样本", f"{len(inputs)} 个：" + ", ".join(p.name for p in inputs))
    info("演示风格", ", ".join(STYLE_CN.get(s, s) for s in args.styles))
    info("输出目录", str(out_dir.resolve()))
    info("停顿模式", "开（边讲边走）" if args.pause else "关（一气呵成）")
    wait(args.pause, "准备就绪，按 Enter 开始演示…")

    t_start = time.time()
    results = []
    for p in inputs:
        try:
            results.append(run_one(p, args.styles, out_dir,
                                   not args.no_plot, args.pause))
        except Exception as e:  # noqa: BLE001
            print(c(f"\n✗ 处理 {p.name} 失败：{type(e).__name__}: {e}",
                    "red", "bold"))

    # 总结
    banner("演示完成  Summary", "magenta")
    info("耗时", f"{time.time() - t_start:.1f}s")
    info("样本数", str(len(results)))
    print()
    header = "  样本            旋律  " + "  ".join(
        f"{s:>9}" for s in args.styles)
    print(c(header, "bold"))
    for r in results:
        row = f"  {r['sample']:<14} {r.get('melody_notes', 0):>4}  "
        row += "  ".join(f"{r['styles'].get(s, {}).get('notes', 0):>9}"
                         for s in args.styles)
        print(row)
    print()
    info("全部产物", str(out_dir.resolve()))
    print(c("\n  ✓ 演示流程结束。每个样本目录下含 4 风格 WAV/MID + 卷帘 PNG。",
            "green", "bold"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
