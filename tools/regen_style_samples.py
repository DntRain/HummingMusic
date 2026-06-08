"""
tools/regen_style_samples.py - week13 风格改进 A/B 重生成

对 reports/week11/ly/ 的 5 个样本，用新 style_postprocess.stylize
重新生成 4 风格 MIDI + WAV，输出到 reports/week13/dc/style_ab/。

不修改 config.yaml，直接调 fluidsynth CLI（绕过 default.sf2 缺失）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("FLUID_NO_AUDIO_DRIVERS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pretty_midi  # noqa: E402

from src.style_postprocess import stylize  # noqa: E402

SF2 = "/usr/share/soundfonts/FluidR3_GM.sf2"
STYLES = ["pop", "jazz", "classical", "folk"]
SRC_DIR = ROOT / "reports/week11/ly"
OUT_DIR = ROOT / "reports/week13/dc/style_ab"


def render_with_fluidsynth(mid_path: Path, wav_path: Path) -> tuple[bool, str]:
    try:
        # fluidsynth 2.5+ 要求选项在 SF2 之前
        r = subprocess.run(
            ["fluidsynth", "-ni", "-F", str(wav_path), "-r", "44100",
             SF2, str(mid_path)],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0, r.stderr[-200:] if r.returncode else ""
    except Exception as e:
        return False, str(e)


def main() -> int:
    if not Path(SF2).exists():
        print(f"❌ soundfont not found: {SF2}")
        return 1

    samples = sorted(d for d in SRC_DIR.iterdir() if d.is_dir())
    print(f"找到 {len(samples)} 个样本，输出至 {OUT_DIR}\n")

    for sample in samples:
        raw_mid = sample / "melody_raw.mid"
        if not raw_mid.exists():
            print(f"⚠️  跳过 {sample.name}：缺 melody_raw.mid")
            continue
        out_dir = OUT_DIR / sample.name
        out_dir.mkdir(parents=True, exist_ok=True)

        midi = pretty_midi.PrettyMIDI(str(raw_mid))
        in_n = sum(len(i.notes) for i in midi.instruments)
        print(f"▶ {sample.name}  in={in_n} notes")

        for st in STYLES:
            out = stylize(midi, st, tempo=120.0)
            tracks = {i.name: len(i.notes) for i in out.instruments}
            total = sum(tracks.values())
            mid_path = out_dir / f"{st}.mid"
            wav_path = out_dir / f"{st}.wav"
            out.write(str(mid_path))
            ok, err = render_with_fluidsynth(mid_path, wav_path)
            mark = "✅" if ok else "❌"
            wav_kb = wav_path.stat().st_size // 1024 if wav_path.exists() else 0
            print(f"   {mark} {st:10}  total={total:>3}  wav={wav_kb}KB  tracks={tracks}")
            if not ok:
                print(f"      err: {err}")
        print()

    print(f"📂 全部输出: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
