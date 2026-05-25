"""
tools/gen_bulk_transfer.py — 批量从 TEST 集生成 N 个样本 × 4 风格 的迁移示例

流程：
    1. 从 TEST split 抽样 candidates 个 key
    2. 对每个候选跑 BiLSTM-CRF 量化，筛出音符数在 [min_notes, max_notes] 的前 N 个
    3. 对每个选中样本 × 4 风格走 VQ-VAE → style_postprocess → FluidSynth → WAV
    4. 输出目录结构：
        out_dir/
          {key1}/
            original_humming.wav   ← 原始哼唱
            gt.mid                 ← ground-truth MIDI（如果可获取）
            melody_raw.mid         ← 量化后的原始旋律
            pop.{mid,wav}
            jazz.{mid,wav}
            classical.{mid,wav}
            folk.{mid,wav}
          {key2}/ ... ...

用法：
    PYTHONPATH=. python tools/gen_bulk_transfer.py --n 5 --out results/bulk_transfer
"""

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.gen_demo import (
    CKPT_PATH,
    STYLES,
    load_quantizer,
    load_vqvae,
    quantize,
    render_wav,
    style_transfer,
)
from src.style_postprocess import stylize

DEFAULT_SPLIT_JSON = "/run/media/DontRain/DATA_NANO/HumTrans/train_valid_test_keys.json"
DEFAULT_FEAT_DIR   = "data/features_crepe"
DEFAULT_WAV_DIR    = "/run/media/DontRain/DATA_NANO/HumTrans/all_wav/wav_data_sync_with_midi"
DEFAULT_MIDI_DIR   = "/run/media/DontRain/DATA_NANO/HumTrans/midi_data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--feat-dir",   default=DEFAULT_FEAT_DIR)
    parser.add_argument("--wav-dir",    default=DEFAULT_WAV_DIR)
    parser.add_argument("--midi-dir",   default=DEFAULT_MIDI_DIR)
    parser.add_argument("--n",          type=int, default=5)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--min-notes",  type=int, default=8)
    parser.add_argument("--max-notes",  type=int, default=30)
    parser.add_argument("--seed",       type=int, default=2026)
    parser.add_argument("--out",        default="results/bulk_transfer")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: %s", device)

    with open(args.split_json, "r", encoding="utf-8") as f:
        test_keys = json.load(f)["TEST"]

    rng = random.Random(args.seed)
    candidates = rng.sample(test_keys, k=min(args.candidates, len(test_keys)))

    q_model = load_quantizer(CKPT_PATH, device)
    selected: list[tuple[str, object, int]] = []
    for key in candidates:
        if len(selected) >= args.n:
            break
        feat_path = Path(args.feat_dir) / f"{key}.npy"
        if not feat_path.exists():
            logger.info("[跳过] %s 特征文件不存在", key)
            continue
        try:
            pm = quantize(str(feat_path), q_model, device)
        except Exception as e:
            logger.warning("[失败] %s 量化报错: %s", key, e)
            continue
        n_notes = sum(len(i.notes) for i in pm.instruments)
        if args.min_notes <= n_notes <= args.max_notes:
            selected.append((key, pm, n_notes))
            logger.info("[选中] %s: %d notes", key, n_notes)
        else:
            logger.info("[跳过] %s: %d notes 不在 [%d, %d]",
                        key, n_notes, args.min_notes, args.max_notes)

    if not selected:
        logger.error("没有符合条件的样本，退出")
        sys.exit(1)

    logger.info("最终选中 %d 个样本", len(selected))

    vq, decoders, cfg = load_vqvae(device)

    for key, melody_pm, n_notes in selected:
        sub_dir = out_dir / key
        sub_dir.mkdir(parents=True, exist_ok=True)

        melody_pm.write(str(sub_dir / "melody_raw.mid"))

        wav_src = Path(args.wav_dir) / f"{key}.wav"
        if wav_src.exists():
            shutil.copy(wav_src, sub_dir / "original_humming.wav")
        gt_src = Path(args.midi_dir) / f"{key}.mid"
        if gt_src.exists():
            shutil.copy(gt_src, sub_dir / "gt.mid")

        for style in STYLES:
            if style not in decoders:
                logger.warning("[%s] decoder 缺失，跳过", style)
                continue
            styled_pm = style_transfer(melody_pm, style, vq, decoders, cfg, device)
            try:
                est_tempo = float(melody_pm.estimate_tempo())
                if not (40.0 < est_tempo < 240.0):
                    est_tempo = 120.0
            except Exception:
                est_tempo = 120.0
            final_pm = stylize(styled_pm, style, tempo=est_tempo)
            mid_path = str(sub_dir / f"{style}.mid")
            wav_path = str(sub_dir / f"{style}.wav")
            final_pm.write(mid_path)
            render_wav(final_pm, wav_path)
        logger.info("✅ %s 完成 (raw=%d notes, 4 风格)", key, n_notes)

    logger.info("全部完成！输出目录: %s", out_dir)


if __name__ == "__main__":
    main()
