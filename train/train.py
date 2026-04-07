"""
train/train.py - BiLSTM-CRF 训练脚本

用法示例：
    # 第一步：预提取特征（只需跑一次）
    python -m train.extract_features --data_root /run/media/DontRain/DATA_NANO/HumTrans

    # 第二步：训练
    python -m train.train \\
        --data_root /run/media/DontRain/DATA_NANO/HumTrans \\
        --feat_dir data/features \\
        --max_train 2000 \\
        --epochs 30 \\
        --lr 1e-3 \\
        --batch_size 32 \\
        --save_dir models/quantizer
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train.dataset import HumTransDataset, collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CRF NLL Loss
# ──────────────────────────────────────────────

def crf_nll_loss(
    model: nn.Module,
    emissions: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算 CRF 负对数似然损失。

    使用前向算法计算配分函数，再减去真实路径分数。

    Args:
        model: BiLSTMCRF 实例。
        emissions: shape=(B, T, num_tags)。
        labels: shape=(B, T)，BIO 标签。
        mask: shape=(B, T)，有效帧为 True。

    Returns:
        Tensor: 标量损失。
    """
    crf = model.crf
    B, T, K = emissions.shape

    # 真实路径分数
    score = crf.start_transitions[labels[:, 0]]  # (B,)
    score += emissions[torch.arange(B), 0, labels[:, 0]]

    for t in range(1, T):
        active = mask[:, t]  # (B,)
        transition = crf.transitions[labels[:, t - 1], labels[:, t]]  # (B,)
        emit = emissions[torch.arange(B), t, labels[:, t]]             # (B,)
        score += (transition + emit) * active.float()

    # end transitions（取最后一个有效帧的标签）
    seq_lengths = mask.long().sum(dim=1) - 1  # (B,)
    last_tags = labels[torch.arange(B), seq_lengths]
    score += crf.end_transitions[last_tags]

    # 配分函数（前向算法）
    log_Z = _forward_algorithm(crf, emissions, mask)

    loss = (log_Z - score).mean()
    return loss


def _forward_algorithm(
    crf: nn.Module,
    emissions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    前向算法计算 log 配分函数。

    Returns:
        Tensor: shape=(B,)，每个样本的 log Z。
    """
    B, T, K = emissions.shape

    alpha = crf.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, K)

    for t in range(1, T):
        active = mask[:, t].unsqueeze(-1)  # (B, 1)
        # (B, K, 1) + (K, K) + (B, 1, K) -> (B, K, K)
        trans_scores = (
            alpha.unsqueeze(2)
            + crf.transitions.unsqueeze(0)
            + emissions[:, t].unsqueeze(1)
        )
        new_alpha = torch.logsumexp(trans_scores, dim=1)  # (B, K)
        alpha = torch.where(active.bool(), new_alpha, alpha)

    alpha += crf.end_transitions.unsqueeze(0)
    return torch.logsumexp(alpha, dim=1)  # (B,)


# ──────────────────────────────────────────────
# 评估
# ──────────────────────────────────────────────

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    在验证集上计算 Note Accuracy（macro recall）。

    Returns:
        float: 平均 Note Accuracy。
    """
    import pretty_midi
    from src.quantizer import _bio_to_notes, _notes_to_midi

    model.eval()
    results = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)   # (B, T, 4)
            mask = batch["mask"].to(device)            # (B, T)
            keys = batch["keys"]

            tag_seqs = model(features, mask)           # list[list[int]]

            for i, tags in enumerate(tag_seqs):
                n = batch["n_frames"][i]
                feat_np = batch["features"][i, :n].numpy()
                # 特征第0列是 midi_note（连续值）
                midi_notes_cont = feat_np[:, 0].astype(float)
                # NaN 处理：is_valid=0 的帧视为 NaN
                valid_mask = feat_np[:, 1] > 0
                midi_notes_cont = midi_notes_cont.copy()
                midi_notes_cont[~valid_mask] = float("nan")

                time_arr = (
                    torch.arange(n).float() * 0.01
                ).numpy()  # 10ms per frame

                notes = _bio_to_notes(tags[:n], time_arr, midi_notes_cont)
                pred_midi = _notes_to_midi(notes, bpm=120.0)

                gt_path = Path(dataloader.dataset.midi_dir) / f"{keys[i]}.mid"
                gt_midi = pretty_midi.PrettyMIDI(str(gt_path))
                results.append((pred_midi, gt_midi))

    from train.metrics import evaluate_dataset as eval_ds
    metrics = eval_ds(results)
    return metrics.note_accuracy


# ──────────────────────────────────────────────
# 训练主循环
# ──────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("使用设备: %s", device)

    # 数据集
    data_root = Path(args.data_root)
    split_json = str(data_root / "train_valid_test_keys.json")
    wav_dir = str(data_root / "all_wav" / "wav_data_sync_with_midi")
    midi_dir = str(data_root / "midi_data")
    feat_dir = args.feat_dir if args.feat_dir else None

    train_ds = HumTransDataset(
        split_json=split_json,
        split="TRAIN",
        wav_dir=wav_dir,
        midi_dir=midi_dir,
        feat_dir=feat_dir,
        max_samples=args.max_train,
    )
    valid_ds = HumTransDataset(
        split_json=split_json,
        split="VALID",
        wav_dir=wav_dir,
        midi_dir=midi_dir,
        feat_dir=feat_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    logger.info("训练集: %d 条  验证集: %d 条", len(train_ds), len(valid_ds))

    # 模型
    from src.quantizer import BiLSTMCRF
    model = BiLSTMCRF(
        input_dim=4,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型参数量: %d", total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ── 训练 ──
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            # LSTM + FC → emissions
            lstm_out, _ = model.lstm(features)
            lstm_out = model.dropout(lstm_out)
            emissions = model.fc(lstm_out)   # (B, T, num_tags)

            loss = crf_nll_loss(model, emissions, labels, mask)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0

        # ── 验证 ──
        val_acc = evaluate(model, valid_loader, device)
        scheduler.step(val_acc)

        logger.info(
            "Epoch %2d/%d | loss=%.4f | val_acc=%.4f | lr=%.2e | %.1fs",
            epoch, args.epochs,
            avg_loss, val_acc,
            optimizer.param_groups[0]["lr"],
            elapsed,
        )

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = save_dir / "bilstm_crf.pt"
            torch.save(model.state_dict(), str(ckpt_path))
            logger.info("  => 保存最优模型 (acc=%.4f): %s", best_acc, ckpt_path)

    logger.info("训练完成。最优验证集 Note Accuracy: %.4f", best_acc)


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BiLSTM-CRF 量化器训练")
    parser.add_argument(
        "--data_root",
        default="/run/media/DontRain/DATA_NANO/HumTrans",
        help="HumTrans 数据集根目录",
    )
    parser.add_argument(
        "--feat_dir",
        default="data/features",
        help="预提取特征目录（None 则实时提取）",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=2000,
        help="训练集样本上限（默认2000）",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--save_dir",
        default="models/quantizer",
        help="模型权重保存目录",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
