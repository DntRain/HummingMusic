"""
train/evaluate.py - 量化器评估脚本

分别评估 RoundingBaselineQuantizer 和 BiLSTM-CRF 在验证集上的 Note Accuracy，
输出对比报告。

用法：
    # 只跑 baseline
    python -m train.evaluate --mode baseline

    # 只跑 BiLSTM-CRF（需要权重）
    python -m train.evaluate --mode model --ckpt models/quantizer/bilstm_crf.pt

    # 两者都跑并对比
    python -m train.evaluate --mode both --ckpt models/quantizer/bilstm_crf.pt
"""

import argparse
import logging
from pathlib import Path

import pretty_midi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def eval_baseline(valid_ds) -> float:
    """在验证集上评估 RoundingBaselineQuantizer。"""
    from src.quantizer import RoundingBaselineQuantizer
    from train.metrics import evaluate_dataset

    quantizer = RoundingBaselineQuantizer()
    results = []

    for i in range(len(valid_ds)):
        item = valid_ds[i]
        key = item["key"]
        feat = item["features"].numpy()
        n = item["n_frames"]

        import numpy as np
        time_arr = np.arange(n) * 0.01

        pitch_data = {
            "time": time_arr,
            "frequency": _midi_to_freq(feat[:n, 0], feat[:n, 1]),
            "confidence": feat[:n, 2],
            "bpm": 120.0,
        }
        pred_midi = quantizer(pitch_data)

        gt_path = Path(valid_ds.midi_dir) / f"{key}.mid"
        gt_midi = pretty_midi.PrettyMIDI(str(gt_path))
        results.append((pred_midi, gt_midi))

        if (i + 1) % 100 == 0:
            logger.info("Baseline 进度: %d/%d", i + 1, len(valid_ds))

    metrics = evaluate_dataset(results)
    return metrics


def eval_model(valid_ds, ckpt_path: str) -> float:
    """在验证集上评估 BiLSTM-CRF 模型。"""
    import torch
    from torch.utils.data import DataLoader
    from train.dataset import collate_fn
    from train.train import evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.quantizer import BiLSTMCRF
    model = BiLSTMCRF(input_dim=4, hidden_size=128, num_layers=2, dropout=0.0)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.to(device)

    loader = DataLoader(
        valid_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    acc = evaluate(model, loader, device)
    return acc


def _midi_to_freq(midi_notes, valid_mask):
    """将 MIDI 编号转回 Hz（invalid 帧置为 NaN）。"""
    import numpy as np  # noqa: F401 – needed for array ops
    freq = 440.0 * (2.0 ** ((midi_notes - 69.0) / 12.0))
    freq[valid_mask < 0.5] = float("nan")
    return freq


def main() -> None:
    parser = argparse.ArgumentParser(description="量化器 Note Accuracy 评估")
    parser.add_argument(
        "--data_root",
        default="/run/media/DontRain/DATA_NANO/HumTrans",
    )
    parser.add_argument("--feat_dir", default="data/features")
    parser.add_argument(
        "--mode",
        choices=["baseline", "model", "both"],
        default="both",
    )
    parser.add_argument(
        "--ckpt",
        default="models/quantizer/bilstm_crf.pt",
        help="BiLSTM-CRF 权重路径",
    )
    parser.add_argument("--split", default="VALID")
    args = parser.parse_args()

    from train.dataset import HumTransDataset
    data_root = Path(args.data_root)

    valid_ds = HumTransDataset(
        split_json=str(data_root / "train_valid_test_keys.json"),
        split=args.split,
        wav_dir=str(data_root / "all_wav" / "wav_data_sync_with_midi"),
        midi_dir=str(data_root / "midi_data"),
        feat_dir=args.feat_dir,
    )

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"评估集: {args.split}  样本数: {len(valid_ds)}")
    print(sep)

    baseline_acc = model_acc = None

    if args.mode in ("baseline", "both"):
        logger.info("评估 RoundingBaselineQuantizer...")
        metrics = eval_baseline(valid_ds)
        baseline_acc = metrics.note_accuracy
        print(f"\n[Baseline] {metrics}")

    if args.mode in ("model", "both"):
        if not Path(args.ckpt).exists():
            logger.error("模型权重不存在: %s", args.ckpt)
        else:
            logger.info("评估 BiLSTM-CRF (%s)...", args.ckpt)
            model_acc = eval_model(valid_ds, args.ckpt)
            print(f"\n[BiLSTM-CRF] Note Accuracy: {model_acc:.4f}")

    if baseline_acc is not None and model_acc is not None:
        delta = model_acc - baseline_acc
        print(f"\n{sep}")
        print(f"Δ (model - baseline) = {delta:+.4f}")
        target = 0.05
        status = "✓ 达标" if delta >= target else f"✗ 未达标 (目标 +{target:.0%})"
        print(f"目标: >= baseline + {target:.0%}  ->  {status}")
        print(f"{sep}\n")


if __name__ == "__main__":
    main()
