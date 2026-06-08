"""
tools/extract_style_vectors.py - 提取各风格 VQ 参考向量

对每种风格的 MIDI 目录，用训练好的 VQ-VAE 编码所有片段，
取平均潜在向量作为该风格的参考向量，保存为 .npy 文件。

用法：
    python tools/extract_style_vectors.py
"""

import logging
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import yaml

from src.style_transfer import StyleVQVAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

with open("config.yaml", "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

ST = _cfg["style_transfer"]
PITCH_LOW  = ST["pitch_low"]
PITCH_HIGH = ST["pitch_high"]
FRAME_RATE = ST["frame_rate"]
SEG_LEN    = 128
STEP       = SEG_LEN // 2

STYLE_DIRS = {
    "pop":       Path(_cfg["data"]["midi_pop_dir"]),
    "jazz":      Path(_cfg["data"]["midi_jazz_dir"]),
    "classical": Path(_cfg["data"]["midi_classical_dir"]),
    "folk":      Path(_cfg["data"]["midi_folk_dir"]),
}


def extract_segments(path: Path) -> list[np.ndarray]:
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
        roll = midi.get_piano_roll(fs=FRAME_RATE)[PITCH_LOW:PITCH_HIGH, :]
        roll = (roll > 0).astype(np.float32)
        T = roll.shape[1]
        if T < SEG_LEN:
            return []
        segs = []
        for start in range(0, T - SEG_LEN + 1, STEP):
            seg = roll[:, start:start + SEG_LEN]
            if seg.sum() > 4:
                segs.append(seg)
        return segs
    except Exception as e:
        logger.debug("跳过 %s: %s", path.name, e)
        return []


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: %s", device)

    ckpt = Path(ST["model_path"])
    if not ckpt.exists():
        logger.error("模型不存在: %s", ckpt)
        return

    model = StyleVQVAE(
        in_channels=ST["in_channels"],
        codebook_size=ST["codebook_size"],
        embedding_dim=ST["embedding_dim"],
    ).to(device)

    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.eval()
    logger.info("VQ-VAE 加载完成: %s", ckpt)

    out_dir = Path(ST["style_vectors_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for style, midi_dir in STYLE_DIRS.items():
        files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
        logger.info("[%s] %d 个 MIDI 文件", style, len(files))

        all_z: list[np.ndarray] = []
        for path in files:
            segs = extract_segments(path)
            if not segs:
                continue
            batch = torch.from_numpy(np.stack(segs)).to(device)  # (N, 48, 128)
            with torch.no_grad():
                z = model.encoder(batch)  # (N, D, T')
                z_mean = z.mean(dim=2)    # (N, D)
            all_z.append(z_mean.cpu().numpy())

        if not all_z:
            logger.warning("[%s] 没有有效片段，跳过", style)
            continue

        style_vec = np.concatenate(all_z, axis=0).mean(axis=0)  # (D,)
        out_path = out_dir / f"{style}_vector.npy"
        np.save(str(out_path), style_vec)
        logger.info("[%s] 参考向量已保存 -> %s  (dim=%d)", style, out_path, style_vec.shape[0])

    logger.info("全部完成。")


if __name__ == "__main__":
    main()
