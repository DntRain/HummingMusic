"""
train/train_vqvae.py - VQ-VAE 风格迁移模型训练脚本

在 HumTrans MIDI 数据上训练 StyleVQVAE，目标是学习旋律的紧凑离散表示。

用法：
    python -m train.train_vqvae \\
        --midi_dir /run/media/DontRain/DATA_NANO/HumTrans/midi_data \\
        --split_json /run/media/DontRain/DATA_NANO/HumTrans/train_valid_test_keys.json \\
        --epochs 50 \\
        --save_dir models/style_transfer
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.style_transfer import StyleVQVAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# 训练超参数
FRAME_RATE   = 32     # 16分音符分辨率下 120BPM 对应 32fps
SEG_LEN      = 128    # 每个训练片段帧数（须为8的倍数，对应3层stride-2下采样）
PITCH_LOW    = 36     # 使用的 MIDI 音高范围低限（C2）
PITCH_HIGH   = 84     # 使用的 MIDI 音高范围高限（C6）
N_PITCHES    = PITCH_HIGH - PITCH_LOW  # 48


class MidiSegmentDataset(Dataset):
    """
    从 MIDI 文件中提取固定长度的 piano roll 片段。

    每个 MIDI 文件按 SEG_LEN 帧分割，二值化（有音符=1，无音符=0）后作为训练样本。
    """

    def __init__(
        self,
        midi_dir: str,
        split_json: str,
        split: str = "TRAIN",
        seg_len: int = SEG_LEN,
        frame_rate: int = FRAME_RATE,
    ) -> None:
        import json
        self.seg_len = seg_len
        self.frame_rate = frame_rate

        with open(split_json, "r", encoding="utf-8") as f:
            keys = json.load(f)[split.upper()]

        midi_dir = Path(midi_dir)
        self.segments: list[np.ndarray] = []

        for key in keys:
            path = midi_dir / f"{key}.mid"
            if not path.exists():
                continue
            try:
                segs = self._extract_segments(path)
                self.segments.extend(segs)
            except Exception as e:
                logger.debug("跳过 %s: %s", key, e)

        logger.info("MidiSegmentDataset [%s]: %d 条片段（来自 %d 个文件）",
                    split, len(self.segments), len(keys))

    def _extract_segments(self, path: Path) -> list[np.ndarray]:
        midi = pretty_midi.PrettyMIDI(str(path))
        # 以固定帧率提取 piano roll，取目标音高范围
        roll = midi.get_piano_roll(fs=self.frame_rate)  # (128, T)
        roll = roll[PITCH_LOW:PITCH_HIGH, :]              # (48, T)

        # 二值化
        roll = (roll > 0).astype(np.float32)

        T = roll.shape[1]
        if T < self.seg_len:
            return []

        # 滑动窗口，步长 seg_len//2（50% 重叠）
        step = self.seg_len // 2
        segs = []
        for start in range(0, T - self.seg_len + 1, step):
            seg = roll[:, start:start + self.seg_len]
            # 过滤空片段（全零）
            if seg.sum() > 4:
                segs.append(seg)
        return segs

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.segments[idx])  # (48, 128)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    # 数据集
    train_ds = MidiSegmentDataset(
        midi_dir=args.midi_dir,
        split_json=args.split_json,
        split="TRAIN",
    )
    valid_ds = MidiSegmentDataset(
        midi_dir=args.midi_dir,
        split_json=args.split_json,
        split="VALID",
    )

    if len(train_ds) == 0:
        logger.error("训练集为空，请检查 midi_dir 和 split_json 路径")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # 模型（输入通道 = N_PITCHES = 48）
    model = StyleVQVAE(
        in_channels=N_PITCHES,
        codebook_size=512,
        embedding_dim=64,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("模型参数量: %d", total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── 训练 ──
        model.train()
        train_recon = train_vq = 0.0

        for batch in train_loader:
            x = batch.to(device)              # (B, 48, 128)
            recon, _, vq_loss = model(x)

            # 重建用 BCE（piano roll 已二值化）
            recon_loss = nn.functional.binary_cross_entropy(recon, x)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_recon += recon_loss.item()
            train_vq    += vq_loss.item()

        train_recon /= len(train_loader)
        train_vq    /= len(train_loader)

        # ── 验证 ──
        model.eval()
        val_recon = val_vq = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                x = batch.to(device)
                recon, _, vq_loss = model(x)
                recon_loss = nn.functional.binary_cross_entropy(recon, x)
                val_recon += recon_loss.item()
                val_vq    += vq_loss.item()

        val_recon /= max(len(valid_loader), 1)
        val_vq    /= max(len(valid_loader), 1)
        val_loss   = val_recon + val_vq

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %3d/%d | train recon=%.4f vq=%.4f | val recon=%.4f vq=%.4f | lr=%.2e",
            epoch, args.epochs,
            train_recon, train_vq,
            val_recon, val_vq, lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, save_dir / "vqvae.pt")
            logger.info("  => 保存最优模型 (val_loss=%.4f): %s/vqvae.pt",
                        val_loss, save_dir)

    logger.info("训练完成。最优验证集 loss: %.4f", best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="VQ-VAE 训练")
    parser.add_argument(
        "--midi_dir",
        default="/run/media/DontRain/DATA_NANO/HumTrans/midi_data",
    )
    parser.add_argument(
        "--split_json",
        default="/run/media/DontRain/DATA_NANO/HumTrans/train_valid_test_keys.json",
    )
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--save_dir",   default="models/style_transfer")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
