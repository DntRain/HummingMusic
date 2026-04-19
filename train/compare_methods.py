"""
train/compare_methods.py - 三种量化方法对比实验

方法：
  1. RoundingBaseline   (pyin 特征)
  2. BiLSTM-CRF v1      (pyin 特征, 2000 训练样本)
  3. BiLSTM-CRF v2      (CREPE 特征, 全量 13080 训练样本)

用法：
    python -m train.compare_methods \\
        --data_root /run/media/DontRain/DATA_NANO/HumTrans \\
        --feat_dir_pyin data/features \\
        --feat_dir_crepe data/features_crepe \\
        --ckpt_v1 models/quantizer/bilstm_crf.pt \\
        --ckpt_v2 models/quantizer_v2/bilstm_crf.pt \\
        --split TEST \\
        --output_dir results/comparison
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pretty_midi
import torch
from torch.utils.data import DataLoader

from train.dataset import HumTransDataset, collate_fn
from train.metrics import NoteMetrics, evaluate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def eval_baseline(ds: HumTransDataset) -> NoteMetrics:
    """评估 RoundingBaselineQuantizer。"""
    from src.quantizer import RoundingBaselineQuantizer

    quantizer = RoundingBaselineQuantizer()
    results = []

    for i in range(len(ds)):
        item = ds[i]
        key = item["key"]
        feat = item["features"].numpy()
        n = item["n_frames"]

        time_arr = np.arange(n) * 0.01
        midi_notes = feat[:n, 0]
        valid_mask = feat[:n, 1]
        freq = 440.0 * (2.0 ** ((midi_notes - 69.0) / 12.0))
        freq[valid_mask < 0.5] = float("nan")

        pitch_data = {
            "time": time_arr,
            "frequency": freq,
            "confidence": feat[:n, 2],
            "bpm": 120.0,
        }
        pred_midi = quantizer(pitch_data)
        gt_midi = pretty_midi.PrettyMIDI(str(ds.midi_dir / f"{key}.mid"))
        results.append((pred_midi, gt_midi))

        if (i + 1) % 100 == 0:
            logger.info("Baseline 进度: %d/%d", i + 1, len(ds))

    return evaluate_dataset(results)


def eval_model_full(ds: HumTransDataset, ckpt_path: str) -> NoteMetrics:
    """评估 BiLSTM-CRF 模型，返回完整 NoteMetrics。"""
    from scipy.signal import find_peaks
    from src.quantizer import BiLSTMCRF, _bio_to_notes, _notes_to_midi

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiLSTMCRF(input_dim=4, hidden_size=128, num_layers=2, dropout=0.0)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    loader = DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            features = batch["features"].to(device)
            lstm_out, _ = model.lstm(features)
            lstm_out = model.dropout(lstm_out)
            emissions = model.fc(lstm_out)         # (B, T, 3)
            b_scores = emissions[:, :, 1].cpu().numpy()

            for i in range(features.shape[0]):
                n = batch["n_frames"][i]
                feat_np = batch["features"][i, :n].numpy()
                midi_notes_cont = feat_np[:, 0].astype(float)
                valid_mask = feat_np[:, 1] > 0
                midi_notes_cont = midi_notes_cont.copy()
                midi_notes_cont[~valid_mask] = float("nan")
                time_arr = np.arange(n, dtype=float) * 0.01

                b_seq = b_scores[i, :n]
                height_thr = b_seq.mean() + 0.5 * b_seq.std()
                peaks, _ = find_peaks(b_seq, distance=20, height=height_thr)

                tags = []
                peak_set = set(peaks.tolist())
                for f in range(n):
                    if f in peak_set:
                        tags.append(1)   # B
                    elif valid_mask[f]:
                        tags.append(2)   # I
                    else:
                        tags.append(0)   # O

                notes = _bio_to_notes(tags, time_arr, midi_notes_cont)
                pred_midi = _notes_to_midi(notes, bpm=120.0)
                key = batch["keys"][i]
                gt_midi = pretty_midi.PrettyMIDI(str(ds.midi_dir / f"{key}.mid"))
                results.append((pred_midi, gt_midi))

            if (batch_idx + 1) % 5 == 0:
                logger.info("  已处理 %d/%d batch", batch_idx + 1, len(loader))

    return evaluate_dataset(results)


def plot_comparison(
    methods: list[str],
    metrics_list: list[NoteMetrics],
    output_path: str,
) -> None:
    """生成对比柱状图，含标注数值。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    metric_labels = ["Note Accuracy\n(Recall)", "Precision", "F1"]
    metric_attrs = ["note_accuracy", "precision", "f1"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    n_methods = len(methods)
    x = np.arange(len(metric_labels))
    total_width = 0.7
    width = total_width / n_methods

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (method, metrics) in enumerate(zip(methods, metrics_list)):
        offset = (i - (n_methods - 1) / 2) * width
        values = [getattr(metrics, attr) for attr in metric_attrs]
        bars = ax.bar(
            x + offset, values, width,
            label=method, color=colors[i % len(colors)], alpha=0.88,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Quantizer Comparison on HumTrans TEST Set\n"
        f"({len(metrics_list[0].n_pred if hasattr(metrics_list[0], 'n_pred') else 0)} samples)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=10)
    max_val = max(getattr(m, a) for m in metrics_list for a in metric_attrs)
    ax.set_ylim(0, max(max_val * 1.35, 0.15))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 在图中标注达标情况
    if len(metrics_list) >= 2:
        best_acc = max(m.note_accuracy for m in metrics_list[1:])
        base_acc = metrics_list[0].note_accuracy
        delta = best_acc - base_acc
        color = "green" if delta >= 0.10 else "red"
        ax.axhline(
            base_acc + 0.10,
            color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
        )
        ax.text(
            len(metric_labels) - 0.5, base_acc + 0.10 + 0.005,
            f"目标线 (baseline+10%)",
            ha="right", va="bottom", fontsize=8, color="gray",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("图表保存至: %s", output_path)


def print_summary(
    method_names: list[str],
    metrics_list: list[NoteMetrics],
) -> None:
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"{'方法':<28} {'NoteAcc':>9} {'Precision':>10} {'Recall':>9} {'F1':>8} {'匹配/GT':>10}")
    print(sep)
    for name, m in zip(method_names, metrics_list):
        match_str = f"{m.n_matched}/{m.n_gt}"
        print(
            f"{name:<28} {m.note_accuracy:>9.4f} {m.precision:>10.4f}"
            f" {m.recall:>9.4f} {m.f1:>8.4f} {match_str:>10}"
        )
    print(sep)

    base_acc = metrics_list[0].note_accuracy
    for name, m in zip(method_names[1:], metrics_list[1:]):
        delta = m.note_accuracy - base_acc
        target_met = "✓ 达标" if delta >= 0.10 else "✗ 未达标"
        print(f"Δ ({name} - Baseline) = {delta:+.4f}  [目标 ≥ +10%]  {target_met}")
    print(f"{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="三种量化方法对比实验")
    parser.add_argument("--data_root", default="/run/media/DontRain/DATA_NANO/HumTrans")
    parser.add_argument("--feat_dir_pyin", default="data/features")
    parser.add_argument("--feat_dir_crepe", default="data/features_crepe")
    parser.add_argument("--ckpt_v1", default="models/quantizer/bilstm_crf.pt")
    parser.add_argument("--ckpt_v2", default="models/quantizer_v2/bilstm_crf.pt")
    parser.add_argument("--split", default="TEST")
    parser.add_argument("--output_dir", default="results/comparison")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_json = str(data_root / "train_valid_test_keys.json")
    wav_dir = str(data_root / "all_wav" / "wav_data_sync_with_midi")
    midi_dir = str(data_root / "midi_data")

    ds_pyin = HumTransDataset(
        split_json=split_json, split=args.split,
        wav_dir=wav_dir, midi_dir=midi_dir, feat_dir=args.feat_dir_pyin,
    )
    ds_crepe = HumTransDataset(
        split_json=split_json, split=args.split,
        wav_dir=wav_dir, midi_dir=midi_dir, feat_dir=args.feat_dir_crepe,
    )

    method_names = []
    metrics_list = []
    json_results = {}

    # ── Method 1: Baseline (pyin) ──
    logger.info("=" * 55)
    logger.info("Method 1: RoundingBaseline (pyin)")
    m1 = eval_baseline(ds_pyin)
    method_names.append("Baseline (pyin)")
    metrics_list.append(m1)
    json_results["baseline_pyin"] = _metrics_to_dict(m1)
    logger.info("Baseline: %s", m1)

    # ── Method 2: BiLSTM-CRF v1 (pyin) ──
    if Path(args.ckpt_v1).exists():
        logger.info("=" * 55)
        logger.info("Method 2: BiLSTM-CRF v1 (pyin, 2000 samples)")
        m2 = eval_model_full(ds_pyin, args.ckpt_v1)
        method_names.append("BiLSTM-CRF v1\n(pyin, 2k)")
        metrics_list.append(m2)
        json_results["bilstm_crf_v1_pyin"] = _metrics_to_dict(m2)
        logger.info("BiLSTM-CRF v1: %s", m2)
    else:
        logger.warning("ckpt_v1 不存在: %s，跳过", args.ckpt_v1)

    # ── Method 3: BiLSTM-CRF v2 (CREPE) ──
    if Path(args.ckpt_v2).exists():
        logger.info("=" * 55)
        logger.info("Method 3: BiLSTM-CRF v2 (CREPE, 13080 samples)")
        m3 = eval_model_full(ds_crepe, args.ckpt_v2)
        method_names.append("BiLSTM-CRF v2\n(CREPE, 13k)")
        metrics_list.append(m3)
        json_results["bilstm_crf_v2_crepe"] = _metrics_to_dict(m3)
        logger.info("BiLSTM-CRF v2: %s", m3)
    else:
        logger.warning("ckpt_v2 不存在: %s，跳过（训练未完成？）", args.ckpt_v2)

    # ── 保存 JSON ──
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    logger.info("JSON 结果保存至: %s/results.json", output_dir)

    # ── 打印汇总 ──
    print_summary(method_names, metrics_list)

    # ── 生成图表 ──
    if len(metrics_list) >= 2:
        plot_comparison(
            method_names,
            metrics_list,
            str(output_dir / "comparison.png"),
        )

    logger.info("全部完成。结果目录: %s", output_dir)


def _metrics_to_dict(m: NoteMetrics) -> dict:
    return {
        "note_accuracy": round(m.note_accuracy, 6),
        "precision": round(m.precision, 6),
        "recall": round(m.recall, 6),
        "f1": round(m.f1, 6),
        "n_pred": m.n_pred,
        "n_gt": m.n_gt,
        "n_matched": m.n_matched,
    }


if __name__ == "__main__":
    main()
