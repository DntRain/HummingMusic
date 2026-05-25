"""
tools/classify_midi.py - 按 GM 乐器程序号将 MIDI 分类到 4 种风格目录

用法：
    python tools/classify_midi.py \
        --src /run/media/DontRain/DATA_NANO/lmd_full \
        --n 200
"""

import argparse
import shutil
from pathlib import Path

import mido

# 各风格的 GM 程序号特征权重
STYLE_PROGRAMS = {
    "jazz": {
        range(0, 8):    1,   # 钢琴
        range(16, 24):  2,   # 风琴
        range(56, 60):  3,   # 铜管
        range(64, 68):  5,   # 萨克斯
        range(32, 34):  1,   # 低音
    },
    "classical": {
        range(0, 8):    1,   # 钢琴
        range(40, 56):  5,   # 弦乐
        range(68, 80):  3,   # 木管
        range(56, 64):  2,   # 铜管
    },
    "pop": {
        range(24, 32):  3,   # 电吉他
        range(33, 41):  3,   # 电贝司
        range(0, 8):    1,   # 钢琴
        range(80, 96):  2,   # 合成器
    },
    "folk": {
        range(24, 26):  5,   # 木吉他
        range(21, 24):  4,   # 手风琴
        range(104, 106): 4,  # 班卓琴
        range(40, 42):  3,   # 小提琴
        range(107, 109): 3,  # 曼陀林
    },
}

SKIP_PROGRAMS = set(range(112, 128))


def score_midi(path: Path) -> dict[str, float]:
    """用 mido 快速读取 program_change 消息提取乐器，返回风格得分。"""
    try:
        mid = mido.MidiFile(str(path))
    except Exception:
        return {}

    programs = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == "program_change" and not getattr(msg, "channel", 0) == 9:
                programs.add(msg.program)

    if not programs or programs.issubset(SKIP_PROGRAMS):
        return {}

    scores: dict[str, float] = {}
    for style, rules in STYLE_PROGRAMS.items():
        s = 0.0
        for prog_range, weight in rules.items():
            for p in programs:
                if p in prog_range:
                    s += weight
        scores[style] = s

    return scores


def classify(src_dir: Path, n: int) -> None:
    # classical 已有数据，只需搜 pop/jazz/folk
    dst_dirs = {
        "jazz":  Path("data/midi_jazz"),
        "pop":   Path("data/midi_pop"),
        "folk":  Path("data/midi_folk"),
    }
    for d in dst_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # 统计各目录已有文件数
    counts: dict[str, int] = {s: len(list(dst_dirs[s].glob("*.mid"))) for s in dst_dirs}
    print(f"已有：{ {s: counts[s] for s in dst_dirs} }")

    filelist = Path("/tmp/lmd_files.txt")
    if filelist.exists():
        midi_files = [Path(p.strip()) for p in filelist.read_text().splitlines() if p.strip()]
    else:
        midi_files = list(src_dir.rglob("*.mid"))
    print(f"文件列表：{len(midi_files)} 个，开始逐一分类（每类满 {n} 首即停）...", flush=True)

    processed = 0
    for path in midi_files:
        # 所有风格都满了就停
        if all(counts[s] >= n for s in dst_dirs):
            break

        scores = score_midi(path)
        if not scores:
            continue

        best_style = max(scores, key=lambda s: scores[s])
        if best_style not in dst_dirs:
            continue  # classical 跳过
        best_score = scores[best_style]
        if best_score == 0 or counts[best_style] >= n:
            continue

        dst = dst_dirs[best_style]
        dest_path = dst / path.name
        if not dest_path.exists():
            shutil.copy2(path, dest_path)
            counts[best_style] += 1
            print(f"  [{best_style:4s} {counts[best_style]:3d}/{n}] {path.name}", flush=True)

        processed += 1
        if processed % 5000 == 0:
            print(f"  -- 已扫描 {processed} 个，进度：{counts}", flush=True)

    print(f"\n完成。最终结果：{counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/run/media/DontRain/DATA_NANO/lmd_full")
    parser.add_argument("--n",   type=int, default=200, help="每种风格取多少首")
    args = parser.parse_args()
    classify(Path(args.src), args.n)


if __name__ == "__main__":
    main()
