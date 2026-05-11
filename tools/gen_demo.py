"""
tools/gen_demo.py - 端到端 Pipeline 示例音频生成

流程：
    特征文件(.npy) → BiLSTM-CRF 量化 → 旋律 MIDI
    → VQ-VAE 风格迁移 × 4 → 各风格 MIDI + WAV

用法：
    python tools/gen_demo.py [--feat path/to/feat.npy] [--out results/demo]
"""

import argparse
import sys
import tempfile
import os
import logging
from pathlib import Path

import numpy as np
import pretty_midi
import torch
from scipy.signal import find_peaks
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────
CKPT_PATH  = "models/quantizer_v5/bilstm_crf.pt"
FRAME_STEP = 0.01   # CREPE 帧步长（秒）
SF2_PATH   = "/home/DontRain/Projects/YOLO11n_Furnas/python312/lib/python3.12/site-packages/pretty_midi/TimGM6mb.sf2"
SYNTH_SR   = 44100
STYLES     = ["pop", "jazz", "classical", "folk"]


# ── 量化器推理 ────────────────────────────────────

def load_quantizer(ckpt: str, device: torch.device):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.quantizer import BiLSTMCRF
    model = BiLSTMCRF(input_dim=4, hidden_size=128, num_layers=2, dropout=0.0)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model


def quantize(feat_path: str, model, device) -> pretty_midi.PrettyMIDI:
    from src.quantizer import _bio_to_notes, _notes_to_midi

    feat = np.load(feat_path).astype(np.float32)          # (T, 4)
    n = len(feat)
    x = torch.from_numpy(feat).unsqueeze(0).to(device)    # (1, T, 4)

    with torch.no_grad():
        lstm_out, _ = model.lstm(x)
        lstm_out = model.dropout(lstm_out)
        emissions = model.fc(lstm_out)                     # (1, T, 3)

    b_scores   = emissions[0, :n, 1].cpu().numpy()
    valid_mask = feat[:, 1] > 0
    midi_hz    = feat[:, 0].astype(float)
    midi_disp  = midi_hz.copy()
    midi_disp[~valid_mask] = float("nan")
    time_arr   = np.arange(n) * FRAME_STEP

    height_thr = b_scores.mean() + 1.5 * b_scores.std()
    peaks, _   = find_peaks(b_scores, distance=5, height=height_thr)
    peak_set   = set(peaks.tolist())
    tags = [1 if f in peak_set else (2 if valid_mask[f] else 0) for f in range(n)]

    notes = _bio_to_notes(tags, time_arr, midi_disp)
    pm    = _notes_to_midi(notes, bpm=120.0)
    logger.info("量化完成：%d 个音符", sum(len(i.notes) for i in pm.instruments))
    return pm


# ── 风格迁移 ──────────────────────────────────────

def load_vqvae(device: torch.device):
    import yaml
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["style_transfer"]

    from src.style_transfer import StyleVQVAE, Decoder
    model = StyleVQVAE(
        in_channels=cfg["in_channels"],
        codebook_size=cfg["codebook_size"],
        embedding_dim=cfg["embedding_dim"],
    ).to(device)
    state = torch.load(cfg["model_path"], map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    # 加载各风格独立解码器
    decoders = {}
    vec_dir = Path(cfg["style_vectors_dir"])
    for style in STYLES:
        dec_path = vec_dir / f"decoder_{style}.pt"
        if dec_path.exists():
            dec = Decoder(out_channels=cfg["in_channels"], embedding_dim=64).to(device)
            dec.load_state_dict(torch.load(str(dec_path), map_location=device, weights_only=False))
            dec.eval()
            decoders[style] = dec
    logger.info("VQ-VAE 加载完成，风格解码器：%s", list(decoders.keys()))
    return model, decoders, cfg


def style_transfer(pm: pretty_midi.PrettyMIDI, style: str,
                   model, decoders: dict, cfg: dict, device: torch.device) -> pretty_midi.PrettyMIDI:
    pitch_low  = cfg["pitch_low"]
    pitch_high = cfg["pitch_high"]
    frame_rate = cfg["frame_rate"]

    roll128 = pm.get_piano_roll(fs=frame_rate)
    roll48  = (roll128[pitch_low:pitch_high] > 0).astype(np.float32)

    T = roll48.shape[1]
    if T < 32:
        roll48 = np.pad(roll48, ((0, 0), (0, 32 - T)))
        T = 32
    pad = (8 - T % 8) % 8
    if pad:
        roll48 = np.pad(roll48, ((0, 0), (0, pad)))

    x = torch.from_numpy(roll48).float().unsqueeze(0).to(device)

    with torch.no_grad():
        z_q, _ = model.encode(x)
        recon = decoders[style](z_q)   # 风格专属解码器

    recon48 = (recon.squeeze(0).cpu().numpy()[:, :T] * 127).clip(0, 127)
    recon128 = np.zeros((128, T), dtype=np.float32)
    recon128[pitch_low:pitch_high] = recon48

    # piano roll → MIDI
    try:
        bpm = pm.estimate_tempo()
    except ValueError:
        bpm = 120.0
    frame_dur = 1.0 / frame_rate
    out = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    from src.style_transfer import STYLE_PROGRAMS
    progs = STYLE_PROGRAMS.get(style, STYLE_PROGRAMS["pop"])
    inst = pretty_midi.Instrument(program=progs["melody"], name="melody")

    for pitch in range(128):
        active = recon128[pitch] > 0
        if not np.any(active):
            continue
        changes = np.diff(active.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends   = np.where(changes == -1)[0] + 1
        if active[0]:
            starts = np.concatenate([[0], starts])
        if active[-1]:
            ends = np.concatenate([ends, [len(active)]])
        for s, e in zip(starts, ends):
            vel = int(np.mean(recon128[pitch, s:e]))
            vel = max(1, min(127, vel))
            inst.notes.append(pretty_midi.Note(vel, pitch, s * frame_dur, e * frame_dur))

    out.instruments.append(inst)
    return out


# ── 音频渲染 ──────────────────────────────────────

def render_wav(pm: pretty_midi.PrettyMIDI, wav_path: str) -> None:
    audio = pm.fluidsynth(fs=SYNTH_SR, sf2_path=SF2_PATH)
    if np.abs(audio).max() > 0:
        audio = audio / np.abs(audio).max() * 0.9
    sf.write(wav_path, audio, SYNTH_SR, subtype="PCM_16")
    logger.info("渲染完成: %s", wav_path)


# ── 主流程 ────────────────────────────────────────

def make_test_melody() -> pretty_midi.PrettyMIDI:
    """构造一段简单测试旋律（C大调音阶，8小节）。"""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, name="melody")
    # C大调音阶，在 C4-C5 范围，每音 0.5s
    pitches = [60, 62, 64, 65, 67, 69, 71, 72,
               71, 69, 67, 65, 64, 62, 60, 60]
    for i, p in enumerate(pitches):
        note = pretty_midi.Note(velocity=90, pitch=p,
                                start=i * 0.5, end=(i + 0.9) * 0.5)
        inst.notes.append(note)
    pm.instruments.append(inst)
    return pm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat", default=None,
                        help="输入特征文件路径（不指定则使用内置测试旋律）")
    parser.add_argument("--out",  default="results/demo",
                        help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: %s", device)

    # 1. 量化 or 使用测试旋律
    if args.feat:
        logger.info("Step 1: 量化 %s", args.feat)
        q_model = load_quantizer(CKPT_PATH, device)
        melody_pm = quantize(args.feat, q_model, device)
    else:
        logger.info("Step 1: 使用内置测试旋律")
        melody_pm = make_test_melody()
    melody_pm.write(str(out_dir / "melody_raw.mid"))
    logger.info("旋律音符数: %d", sum(len(i.notes) for i in melody_pm.instruments))

    # 2. 加载 VQ-VAE
    logger.info("Step 2: 加载 VQ-VAE")
    vq_model, decoders, cfg = load_vqvae(device)

    # 3. 各风格迁移 + 渲染
    for style in STYLES:
        if style not in decoders:
            logger.warning("[%s] 解码器缺失，跳过", style)
            continue
        logger.info("Step 3 [%s]: 风格迁移", style)
        styled_pm = style_transfer(melody_pm, style, vq_model, decoders, cfg, device)
        mid_path  = str(out_dir / f"{style}.mid")
        wav_path  = str(out_dir / f"{style}.wav")
        styled_pm.write(mid_path)
        render_wav(styled_pm, wav_path)

    logger.info("完成！输出目录: %s", out_dir)
    for f in sorted(out_dir.iterdir()):
        logger.info("  %s  (%.1f KB)", f.name, f.stat().st_size / 1024)


if __name__ == "__main__":
    main()
