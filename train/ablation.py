"""
train/ablation.py - BiLSTM-CRF v5 消融实验

在 TEST 集（769 样本）上对比 4 个配置：

  1. Full v5 (baseline)            完整 pipeline
  2. w/o CRF                       去掉 CRF Viterbi/peak-detection，直接帧级 argmax
  3. w/o Confidence Filter         不使用 features 的 valid_mask 过滤
  4. w/o Silence Interpolation     模拟未做短静音插值（低置信度帧重新视为静音）

输出指标：Precision / Recall / F1 / Note Accuracy（macro 平均）。

用法：
    python -m train.ablation [--max_samples N]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pretty_midi
import torch
from scipy.signal import find_peaks
from torch.utils.data import DataLoader

from src.quantizer import BiLSTMCRF, _bio_to_notes, _notes_to_midi
from train.dataset import HumTransDataset, collate_fn
from train.metrics import evaluate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────
# 解码策略：每个返回长度为 n 的 BIO tags（list[int]）
# ────────────────────────────────────────────────────────

def decode_full(emissions: np.ndarray, feat: np.ndarray, n: int) -> list[int]:
    """v5 baseline: find_peaks(B 发射分数) + valid_mask 作 I/O 区分。"""
    b_seq = emissions[:n, 1]
    height_thr = b_seq.mean() + 0.5 * b_seq.std()
    peaks, _ = find_peaks(b_seq, distance=20, height=height_thr)
    valid_mask = feat[:n, 1] > 0
    peak_set = set(peaks.tolist())
    return [1 if f in peak_set else (2 if valid_mask[f] else 0) for f in range(n)]


def decode_no_crf(emissions: np.ndarray, feat: np.ndarray, n: int) -> list[int]:
    """去掉 CRF：直接对 emissions 取 argmax，每帧独立预测。"""
    return emissions[:n].argmax(axis=-1).tolist()


def decode_no_conf_filter(emissions: np.ndarray, feat: np.ndarray, n: int) -> list[int]:
    """去掉置信度过滤：valid_mask 用 confidence > 0.1 重算（模拟未做 0.8 阈值过滤）。"""
    b_seq = emissions[:n, 1]
    height_thr = b_seq.mean() + 0.5 * b_seq.std()
    peaks, _ = find_peaks(b_seq, distance=20, height=height_thr)
    valid_mask = feat[:n, 2] > 0.1
    peak_set = set(peaks.tolist())
    return [1 if f in peak_set else (2 if valid_mask[f] else 0) for f in range(n)]


def decode_no_silence_interp(emissions: np.ndarray, feat: np.ndarray, n: int) -> list[int]:
    """去掉短静音插值：把低置信度帧（被插值修复过的可能位置）重新视为静音。"""
    b_seq = emissions[:n, 1]
    height_thr = b_seq.mean() + 0.5 * b_seq.std()
    peaks, _ = find_peaks(b_seq, distance=20, height=height_thr)
    valid_mask = (feat[:n, 1] > 0).copy()
    # 模拟"未插值"：valid=1 但 confidence 较低的帧（可能是被插值修复的）打回 0
    suspect = valid_mask & (feat[:n, 2] < 0.5)
    valid_mask = valid_mask & ~suspect
    peak_set = set(peaks.tolist())
    return [1 if f in peak_set else (2 if valid_mask[f] else 0) for f in range(n)]


DECODERS = {
    "Full v5 (baseline)":          decode_full,
    "w/o CRF":                     decode_no_crf,
    "w/o Confidence Filter":       decode_no_conf_filter,
    "w/o Silence Interpolation":   decode_no_silence_interp,
}


# ────────────────────────────────────────────────────────
# 评估循环
# ────────────────────────────────────────────────────────

def run_all_configs(model, loader, device, decoders: dict):
    """单次 forward，多个 decode 策略共享 emissions/GT，最后分别评估。"""
    model.eval()
    results_per_cfg = {name: [] for name in decoders}
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            features = batch["features"].to(device)
            keys = batch["keys"]
            lstm_out, _ = model.lstm(features)
            lstm_out = model.dropout(lstm_out)
            emissions = model.fc(lstm_out).cpu().numpy()       # (B, T, 3)

            for i in range(features.shape[0]):
                n = batch["n_frames"][i]
                feat_np = batch["features"][i, :n].numpy()

                midi_notes = feat_np[:, 0].astype(float).copy()
                midi_notes[~(feat_np[:, 1] > 0)] = float("nan")
                time_arr = np.arange(n, dtype=float) * 0.01

                gt_path = Path(loader.dataset.midi_dir) / f"{keys[i]}.mid"
                gt = pretty_midi.PrettyMIDI(str(gt_path))

                for name, fn in decoders.items():
                    tags = fn(emissions[i], feat_np, n)
                    notes = _bio_to_notes(tags, time_arr, midi_notes)
                    pred = _notes_to_midi(notes, bpm=120.0)
                    results_per_cfg[name].append((pred, gt))

            if (bi + 1) % 5 == 0:
                logger.info("  已处理 %d/%d batch", bi + 1, len(loader))
    return {name: evaluate_dataset(res) for name, res in results_per_cfg.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/run/media/DontRain/DATA_NANO/HumTrans")
    parser.add_argument("--feat_dir",  default="data/features")
    parser.add_argument("--ckpt",      default="models/quantizer_v5/bilstm_crf.pt")
    parser.add_argument("--split",     default="TEST")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out_json",  default="reports/week10/ablation_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: %s", device)

    data_root = Path(args.data_root)
    ds = HumTransDataset(
        split_json=str(data_root / "train_valid_test_keys.json"),
        split=args.split,
        wav_dir=str(data_root / "all_wav" / "wav_data_sync_with_midi"),
        midi_dir=str(data_root / "midi_data"),
        feat_dir=args.feat_dir,
        max_samples=args.max_samples,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)

    model = BiLSTMCRF(input_dim=4, hidden_size=128, num_layers=2, dropout=0.0)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt)
    model.to(device)

    print(f"\n{'='*70}")
    print(f"消融实验 | split={args.split} | 样本数={len(ds)}")
    print('='*70)

    t0 = time.time()
    metrics_by_cfg = run_all_configs(model, loader, device, DECODERS)
    dt_total = time.time() - t0
    logger.info("全部 4 个 config 共耗时 %.1fs", dt_total)

    all_results: dict[str, dict] = {}
    baseline_acc = None
    for name in DECODERS:
        m = metrics_by_cfg[name]
        all_results[name] = {
            "precision": m.precision, "recall": m.recall,
            "f1": m.f1, "note_accuracy": m.note_accuracy,
            "n_pred": m.n_pred, "n_gt": m.n_gt, "n_matched": m.n_matched,
        }
        if baseline_acc is None:
            baseline_acc = m.note_accuracy
            delta = ""
        else:
            delta = f"  Δ={m.note_accuracy - baseline_acc:+.4f}"
        print(f"\n[{name}]")
        print(f"  P={m.precision:.4f}  R={m.recall:.4f}  F1={m.f1:.4f}  "
              f"NoteAcc={m.note_accuracy:.4f}{delta}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "n_samples": len(ds),
                   "ckpt": args.ckpt, "seconds": dt_total,
                   "results": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
