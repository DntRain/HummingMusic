"""生成 Piano Roll 图例修复前后的对比图。"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pretty_midi
from matplotlib import font_manager
from matplotlib.patches import Patch

matplotlib.use("Agg")

for fp in [
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
]:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def make_demo_midi() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    melody = pretty_midi.Instrument(program=0, name="melody")
    for i, p in enumerate([60, 62, 64, 67, 69, 67, 64, 62, 60]):
        melody.notes.append(pretty_midi.Note(
            velocity=100, pitch=p, start=i * 0.4, end=i * 0.4 + 0.35))
    accomp = pretty_midi.Instrument(program=0, name="accompaniment")
    for i, root in enumerate([48, 53, 55, 48]):
        for offset in (0, 4, 7):
            accomp.notes.append(pretty_midi.Note(
                velocity=70, pitch=root + offset,
                start=i * 0.9, end=i * 0.9 + 0.85))
    pm.instruments.extend([melody, accomp])
    return pm


def plot(midi, out: Path, with_legend: bool, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    for inst in midi.instruments:
        color = "steelblue" if inst.name == "melody" else "coral"
        for n in inst.notes:
            ax.barh(n.pitch, n.end - n.start, left=n.start,
                    height=0.8, alpha=0.7, color=color)
    if with_legend:
        ax.legend(handles=[
            Patch(facecolor="steelblue", alpha=0.7, label="旋律 (melody)"),
            Patch(facecolor="coral", alpha=0.7, label="伴奏 (accompaniment)"),
        ], loc="upper right")
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("MIDI 音高")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    out_dir = Path(__file__).resolve().parent.parent / "reports/week12/ui_screenshots"
    midi = make_demo_midi()
    plot(midi, out_dir / "04_before_piano_roll.png",
         with_legend=False, title="Piano Roll (修复前：无图例)")
    plot(midi, out_dir / "05_after_piano_roll.png",
         with_legend=True, title="Piano Roll (修复后：含图例)")


if __name__ == "__main__":
    main()
