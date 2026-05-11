"""
train/finetune_decoder.py - 冻结编码器，为各风格独立 fine-tune 解码器

流程：
    1. 加载预训练 VQ-VAE，冻结 Encoder + VectorQuantizer
    2. 对每种风格的 MIDI 数据集，单独训练一个 Decoder
    3. 各解码器保存为 models/style_transfer/decoder_{style}.pt

用法：
    python -m train.finetune_decoder
    python -m train.finetune_decoder --styles jazz pop  # 只训练指定风格
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.style_transfer import StyleVQVAE, Decoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PITCH_LOW  = 36
PITCH_HIGH = 84
N_PITCHES  = PITCH_HIGH - PITCH_LOW   # 48
FRAME_RATE = 32
SEG_LEN    = 128
STEP       = SEG_LEN // 2

STYLE_DIRS = {
    "pop":       Path("data/midi_pop"),
    "jazz":      Path("data/midi_jazz"),
    "classical": Path("data/midi_classical"),
    "folk":      Path("data/midi_folk"),
}


class StyleMidiDataset(Dataset):
    def __init__(self, midi_dir: Path) -> None:
        self.segments: list[np.ndarray] = []
        files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
        for path in files:
            try:
                midi = pretty_midi.PrettyMIDI(str(path))
                roll = midi.get_piano_roll(fs=FRAME_RATE)[PITCH_LOW:PITCH_HIGH]
                roll = (roll > 0).astype(np.float32)
                T = roll.shape[1]
                for start in range(0, T - SEG_LEN + 1, STEP):
                    seg = roll[:, start:start + SEG_LEN]
                    if seg.sum() > 4:
                        self.segments.append(seg)
            except Exception as e:
                logger.debug("跳过 %s: %s", path.name, e)
        logger.info("  数据集: %d 片段（来自 %d 文件）", len(self.segments), len(files))

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.segments[idx])


def finetune_style(
    style: str,
    encoder_vq,           # 冻结的 encoder+vq
    pretrained_decoder: Decoder,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    midi_dir = STYLE_DIRS[style]
    logger.info("── [%s] 开始 fine-tune ──", style)

    ds = StyleMidiDataset(midi_dir)
    if len(ds) == 0:
        logger.warning("[%s] 数据集为空，跳过", style)
        return

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    # 从预训练解码器拷贝权重，独立 fine-tune
    decoder = Decoder(out_channels=N_PITCHES, embedding_dim=64).to(device)
    decoder.load_state_dict(pretrained_decoder.state_dict())

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_loss = float("inf")
    save_dir  = Path(args.save_dir)

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        total_loss = 0.0

        for batch in loader:
            x = batch.to(device)                       # (B, 48, 128)

            # 编码（不计算梯度）
            with torch.no_grad():
                z, _, _ = encoder_vq(x)                # z_q: (B, 64, 16)

            # 解码（计算梯度）
            recon = decoder(z)                         # (B, 48, 128)
            loss  = nn.functional.binary_cross_entropy(recon, x)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        lr = optimizer.param_groups[0]["lr"]
        logger.info("[%s] Epoch %2d/%d  loss=%.4f  lr=%.2e",
                    style, epoch, args.epochs, avg_loss, lr)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(decoder.state_dict(), save_dir / f"decoder_{style}.pt")
            logger.info("  => 保存最优解码器 (loss=%.4f)", best_loss)

    logger.info("[%s] 完成，最优 loss=%.4f", style, best_loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default="models/style_transfer/vqvae.pt")
    parser.add_argument("--save_dir",   default="models/style_transfer")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--styles",     nargs="+",  default=list(STYLE_DIRS.keys()))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: %s", device)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # 加载预训练 VQ-VAE
    model = StyleVQVAE(in_channels=N_PITCHES, codebook_size=512, embedding_dim=64).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    # 冻结 encoder + VQ（只保留 decoder 用于初始化）
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.vq.parameters():
        p.requires_grad_(False)

    # encoder_vq: 只做 encode（返回 z_q）
    class EncoderVQ(nn.Module):
        def __init__(self, enc, vq):
            super().__init__()
            self.encoder = enc
            self.vq = vq
        def forward(self, x):
            z = self.encoder(x)
            z_q, indices, vq_loss = self.vq(z)
            return z_q, indices, vq_loss

    encoder_vq = EncoderVQ(model.encoder, model.vq).to(device)

    for style in args.styles:
        if style not in STYLE_DIRS:
            logger.warning("未知风格: %s，跳过", style)
            continue
        finetune_style(style, encoder_vq, model.decoder, device, args)

    logger.info("所有风格 fine-tune 完成。")


if __name__ == "__main__":
    main()
