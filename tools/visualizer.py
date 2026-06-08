"""
tools/visualizer.py - BiLSTM-CRF 量化器交互式可视化工具

用法：
    streamlit run tools/visualizer.py
"""

import os
# Bug-05 修复：在导入 fluidsynth/pretty_midi 之前抑制 ALSA 硬件探测警告
os.environ.setdefault("FLUID_NO_AUDIO_DRIVERS", "1")

import sys
import contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import json
import time
import tempfile
import numpy as np
import pretty_midi
import soundfile as sf
import torch
from scipy.signal import find_peaks
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from train.dataset import HumTransDataset
from src.quantizer import BiLSTMCRF, _bio_to_notes, _notes_to_midi
from src.audio_processing import LowEnergyError, _check_rms_energy
from train.metrics import compute_note_metrics

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
DATA_ROOT   = "/run/media/DontRain/DATA_NANO/HumTrans"
FEAT_DIR    = "data/features_crepe"
CKPT_PATH   = "models/quantizer_v5/bilstm_crf.pt"
SPLIT_JSON  = f"{DATA_ROOT}/train_valid_test_keys.json"
WAV_DIR     = f"{DATA_ROOT}/all_wav/wav_data_sync_with_midi"
MIDI_DIR    = f"{DATA_ROOT}/midi_data"
FRAME_STEP  = 0.01  # 秒/帧
_SF2_CANDIDATES = [
    "/usr/share/soundfonts/FluidR3_GM2-2.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    str(Path(__file__).parent.parent.parent /
        "YOLO11n_Furnas/python312/lib/python3.12/site-packages/pretty_midi/TimGM6mb.sf2"),
]
SF2_PATH    = next((p for p in _SF2_CANDIDATES if Path(p).exists()), None)
SYNTH_SR    = 22050

DEMO_DIR    = Path(__file__).parent.parent / "data" / "demo"
DEMO_WAV    = DEMO_DIR / "example.wav"
DEMO_FEAT   = DEMO_DIR / "example.npy"

REC_DIR     = Path(__file__).parent.parent / "data" / "recordings"


class _LocalAudioFile:
    """把本地音频文件包装成与 UploadedFile 同构的对象。

    录音先写入 data/recordings/rec_<hash>.wav，再对外提供与
    st.file_uploader 返回值一致的 .name / .size / .getbuffer() 接口，
    让 validate / extract_features 继续走上传文件同一套逻辑。
    """

    def __init__(self, save_path: Path):
        self.path = save_path
        self.name = save_path.name
        self.size = save_path.stat().st_size

    def getbuffer(self) -> memoryview:
        return memoryview(self.path.read_bytes())


class _AudioSource:
    """UI 层使用的统一音频输入描述。"""

    def __init__(self, file_obj, label: str, mode_label: str, key_name: str):
        self.file = file_obj
        self.label = label
        self.mode_label = mode_label
        self.key_name = key_name

    @property
    def name(self) -> str:
        return self.file.name

    @property
    def size(self) -> int:
        return getattr(self.file, "size", 0)

    def getbuffer(self) -> memoryview:
        return self.file.getbuffer()


def _save_uploaded_audio_to_disk(uploaded_file) -> _LocalAudioFile:
    """保存 st.audio_input 的录音，并返回 UploadedFile 兼容对象。"""
    REC_DIR.mkdir(parents=True, exist_ok=True)
    raw = uploaded_file.getbuffer().tobytes()
    import hashlib
    digest = hashlib.md5(raw).hexdigest()[:10]
    suffix = Path(getattr(uploaded_file, "name", "")).suffix.lower() or ".wav"
    out_path = REC_DIR / f"rec_{digest}{suffix}"
    if not out_path.exists():
        out_path.write_bytes(raw)
    return _LocalAudioFile(out_path)


def _save_recording_to_disk(recorded_audio) -> _LocalAudioFile | None:
    """把 st.audio_input 返回的录音落盘到 REC_DIR。"""
    if recorded_audio is None:
        return None
    return _save_uploaded_audio_to_disk(recorded_audio)


def load_demo_item() -> dict:
    """把内置演示音频包装成与 HumTransDataset 同结构的 item。"""
    feat = np.load(DEMO_FEAT).astype(np.float32)
    return {
        "features": torch.from_numpy(feat),
        "n_frames": len(feat),
        "key":      "example",
    }


@st.cache_data(show_spinner=False, max_entries=8)
def extract_features_from_bytes(file_bytes: bytes, suffix: str) -> tuple[np.ndarray, bytes]:
    """上传文件 → (特征 ndarray (T,4), 重采样后的 16k mono WAV bytes)。

    用 tempdir 写入原始文件，必要时 ffmpeg 转 16k mono，再调用
    train.extract_features.extract_one（CREPE 优先，pyin 回退）。
    """
    import subprocess
    import librosa
    from train.extract_features import extract_one

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / f"upload{suffix.lower()}"
        src.write_bytes(file_bytes)

        wav16 = td_path / "example.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-ac", "1", "-ar", "16000", "-loglevel", "error", str(wav16)],
            check=True,
        )

        # Bug-04 根因修复：CREPE 前先做能量门控，省 6-10s 空推理
        audio_pcm, _ = librosa.load(str(wav16), sr=16000, mono=True)
        _check_rms_energy(audio_pcm)

        if not extract_one("example", td_path, td_path, force=True):
            raise RuntimeError("特征提取失败（extract_one 返回 False）")
        feat = np.load(td_path / "example.npy").astype(np.float32)
        wav_bytes = wav16.read_bytes()
        return feat, wav_bytes


# ──────────────────────────────────────────────
# 缓存：加载模型（只加载一次）
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMCRF(input_dim=4, hidden_size=128, num_layers=2, dropout=0.0)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def load_dataset(split: str, correct_octave: bool = True):
    import logging
    logging.disable(logging.WARNING)
    ds = HumTransDataset(
        split_json=SPLIT_JSON, split=split,
        wav_dir=WAV_DIR, midi_dir=MIDI_DIR,
        feat_dir=FEAT_DIR,
        correct_octave=correct_octave,
    )
    logging.disable(logging.NOTSET)
    return ds


# ──────────────────────────────────────────────
# MIDI 合成
# ──────────────────────────────────────────────
@contextlib.contextmanager
def _silence_stderr_fd():
    """Bug-05 修复：用 fd 级别重定向吞掉 fluidsynth/libasound 的 C 层 stderr。"""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


@st.cache_data(max_entries=20)
def synthesize_midi(midi_bytes: bytes, instrument_name: str = "Acoustic Grand Piano") -> bytes:
    """将 MIDI bytes 合成为 WAV bytes（用于 st.audio 播放）。"""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        f.write(midi_bytes)
        tmp_path = f.name
    try:
        pm = pretty_midi.PrettyMIDI(tmp_path)
        program = pretty_midi.instrument_name_to_program(instrument_name)
        for inst in pm.instruments:
            if not inst.is_drum:
                inst.program = program
        try:
            if SF2_PATH:
                with _silence_stderr_fd():
                    audio = pm.fluidsynth(fs=SYNTH_SR, sf2_path=SF2_PATH)
            else:
                audio = pm.synthesize(fs=SYNTH_SR)
        except Exception:
            audio = pm.synthesize(fs=SYNTH_SR)
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.9
        buf = io.BytesIO()
        sf.write(buf, audio, SYNTH_SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    finally:
        os.unlink(tmp_path)


def midi_to_bytes(pm: pretty_midi.PrettyMIDI) -> bytes:
    """PrettyMIDI 对象序列化为 bytes。"""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp_path = f.name
    try:
        pm.write(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


# ──────────────────────────────────────────────
# 推理
# ──────────────────────────────────────────────
def run_inference(item: dict, model, device, peak_distance: int, peak_height_sigma: float):
    feat = item["features"].unsqueeze(0).to(device)
    n = item["n_frames"]

    with torch.no_grad():
        lstm_out, _ = model.lstm(feat)
        lstm_out = model.dropout(lstm_out)
        emissions = model.fc(lstm_out)

    b_scores = emissions[0, :n, 1].cpu().numpy()
    i_scores = emissions[0, :n, 2].cpu().numpy()
    o_scores = emissions[0, :n, 0].cpu().numpy()

    feat_np = item["features"][:n].numpy()
    midi_notes_cont = feat_np[:, 0].astype(float)
    valid_mask = feat_np[:, 1] > 0
    confidence = feat_np[:, 2]

    midi_notes_disp = midi_notes_cont.copy()
    midi_notes_disp[~valid_mask] = float("nan")
    time_arr = np.arange(n) * FRAME_STEP

    height_thr = b_scores.mean() + peak_height_sigma * b_scores.std()
    peaks, _ = find_peaks(b_scores, distance=peak_distance, height=height_thr)

    peak_set = set(peaks.tolist())
    tags = [1 if f in peak_set else (2 if valid_mask[f] else 0) for f in range(n)]

    notes = _bio_to_notes(tags, time_arr, midi_notes_disp)
    pred_midi = _notes_to_midi(notes, bpm=120.0)

    return {
        "pred_midi": pred_midi,
        "b_scores": b_scores,
        "i_scores": i_scores,
        "o_scores": o_scores,
        "peaks": peaks,
        "height_thr": height_thr,
        "time_arr": time_arr,
        "midi_notes": midi_notes_disp,
        "valid_mask": valid_mask,
        "confidence": confidence,
        "n": n,
    }


# ──────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────
def plot_piano_roll(pred_midi, gt_midi, duration: float, title: str = ""):
    def get_notes(midi):
        if midi is None:
            return []
        notes = []
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                notes.append((note.pitch, note.start, note.end))
        return notes

    pred_notes = get_notes(pred_midi)
    gt_notes   = get_notes(gt_midi)

    all_pitches = [p for p, _, _ in pred_notes + gt_notes]
    if not all_pitches:
        return None
    p_min = max(min(all_pitches) - 2, 0)
    p_max = min(max(all_pitches) + 3, 127)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    for pitch, start, end in gt_notes:
        rect = mpatches.FancyBboxPatch(
            (start, pitch - 0.45), end - start, 0.9,
            boxstyle="round,pad=0.02",
            facecolor="#2ecc71", edgecolor="#27ae60", linewidth=0.8, alpha=0.55,
        )
        ax.add_patch(rect)

    for pitch, start, end in pred_notes:
        rect = mpatches.FancyBboxPatch(
            (start, pitch - 0.42), end - start, 0.84,
            boxstyle="round,pad=0.02",
            facecolor="none", edgecolor="#3498db", linewidth=1.5, alpha=0.95,
        )
        ax.add_patch(rect)

    ax.set_xlim(0, duration)
    ax.set_ylim(p_min, p_max)
    ax.set_xlabel("时间 (s)", color="white", fontsize=10)
    ax.set_ylabel("MIDI 音高", color="white", fontsize=10)
    ax.tick_params(colors="white")

    yticks = list(range(p_min, p_max + 1, 2))
    yticklabels = [pretty_midi.note_number_to_name(p) for p in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7, color="white")

    ax.grid(axis="x", color="white", alpha=0.1, linestyle="--")

    gt_patch   = mpatches.Patch(facecolor="#2ecc71", alpha=0.55, label="GT（绿色填充）")
    pred_patch = mpatches.Patch(facecolor="none", edgecolor="#3498db",
                                linewidth=1.5, label="预测（蓝色边框）")
    ax.legend(handles=[gt_patch, pred_patch], loc="upper right",
              facecolor="#2a2a4a", labelcolor="white", fontsize=9)

    if title:
        ax.set_title(title, color="white", fontsize=11)

    plt.tight_layout()
    return fig


def plot_scores(result: dict):
    time_arr = result["time_arr"]
    b_scores = result["b_scores"]
    confidence = result["confidence"]
    peaks = result["peaks"]
    height_thr = result["height_thr"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")

    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    ax.plot(time_arr, b_scores, color="#e74c3c", linewidth=0.8, label="B 分数（音符起始）")
    ax.axhline(height_thr, color="yellow", linewidth=0.8, linestyle="--", alpha=0.7,
               label=f"阈值 {height_thr:.2f}")
    if len(peaks):
        ax.scatter(time_arr[peaks], b_scores[peaks], color="yellow",
                   s=25, zorder=5, label=f"峰值 ({len(peaks)}个)")
    ax.set_ylabel("得分", color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", facecolor="#2a2a4a", labelcolor="white", fontsize=8)
    ax.set_title("模型 B 类得分（音符起始检测）", color="white", fontsize=10)
    ax.grid(alpha=0.15, color="white", linestyle="--")

    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    ax2.plot(time_arr, confidence, color="#3498db", linewidth=0.8, alpha=0.8,
             label="CREPE 置信度")
    ax2.set_ylabel("置信度", color="white", fontsize=9)
    ax2.set_xlabel("时间 (s)", color="white", fontsize=9)
    ax2.tick_params(colors="white")
    ax2.legend(loc="upper right", facecolor="#2a2a4a", labelcolor="white", fontsize=8)
    ax2.grid(alpha=0.15, color="white", linestyle="--")

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# 错误检测常量
# ──────────────────────────────────────────────
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MIN_DURATION_SEC   = 1.0
INFERENCE_TIMEOUT  = 10.0

# ──────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="HummingMusic · 哼唱量化与风格迁移",
    page_icon="🎵",
    layout="wide",
)


def validate_uploaded_audio(uploaded_file) -> tuple[bool, str, float | None, int | None]:
    """返回 (ok, error_msg, duration_sec, sample_rate)。用 ffprobe 探测元数据。"""
    if uploaded_file is None:
        return False, "", None, None

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTS:
        return False, (f"格式错误：不支持 {suffix} 格式，请上传 "
                       f"{' / '.join(sorted(ALLOWED_AUDIO_EXTS))} 文件。"), None, None

    import subprocess
    import json as _json
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(uploaded_file.getbuffer())
            tmp = f.name
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_entries", "format=duration:stream=sample_rate,codec_type",
             tmp],
            capture_output=True, text=True, check=True,
        )
        info = _json.loads(probe.stdout)
        astreams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
        if not astreams:
            return False, "格式错误：文件中未找到音频流。", None, None
        duration = float(info.get("format", {}).get("duration", 0.0))
        sr = int(astreams[0].get("sample_rate", 0))
    except subprocess.CalledProcessError as e:
        return False, f"格式错误：文件无法解析（{e.stderr.strip() or e}）", None, None
    except Exception as e:
        return False, f"格式错误：文件无法解析（{e}）", None, None
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

    if duration < MIN_DURATION_SEC:
        return False, (f"音频过短：时长 {duration:.2f}s，最短需 "
                       f"{MIN_DURATION_SEC}s。请上传更长的录音。"), None, None

    return True, "", duration, sr


def run_inference_with_timeout(item, model, device, peak_distance, peak_height_sigma):
    """带超时检测的推理包装，超时返回 None。"""
    t0 = time.time()
    result = run_inference(item, model, device, peak_distance, peak_height_sigma)
    elapsed = time.time() - t0
    if elapsed > INFERENCE_TIMEOUT:
        return None, elapsed
    return result, elapsed


# ──────────────────────────────────────────────
# 风格迁移辅助
# ──────────────────────────────────────────────

# 情感旋钮：bpm_scale 作用在 stylize 的 tempo（影响所有节拍/和弦时长），
# vel_* 在 stylize 输出后对每个 note.velocity 做仿射缩放。
EMOTION_PARAMS: dict[str, dict] = {
    "neutral": {"bpm_scale": 1.00, "vel_scale": 1.00, "vel_offset":   0, "vel_range": 1.00},
    "happy":   {"bpm_scale": 1.10, "vel_scale": 1.05, "vel_offset":  +8, "vel_range": 1.15},
    "sad":     {"bpm_scale": 0.85, "vel_scale": 0.90, "vel_offset": -12, "vel_range": 0.85},
    "calm":    {"bpm_scale": 0.85, "vel_scale": 0.92, "vel_offset":  -6, "vel_range": 0.80},
    "intense": {"bpm_scale": 1.18, "vel_scale": 1.15, "vel_offset": +12, "vel_range": 1.30},
}


def _apply_emotion_velocity(pm: pretty_midi.PrettyMIDI, emotion: str) -> None:
    """围绕中值 64 做 range 缩放，再乘 scale 加 offset。原地修改。"""
    if emotion == "neutral":
        return
    p = EMOTION_PARAMS[emotion]
    for inst in pm.instruments:
        for note in inst.notes:
            v = 64 + (note.velocity - 64) * p["vel_range"]
            v = v * p["vel_scale"] + p["vel_offset"]
            note.velocity = max(1, min(127, int(round(v))))


@st.cache_resource
def load_vqvae():
    import yaml
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["style_transfer"]
    from src.style_transfer import StyleVQVAE, Decoder
    vq = StyleVQVAE(
        in_channels=cfg["in_channels"],
        codebook_size=cfg["codebook_size"],
        embedding_dim=cfg["embedding_dim"],
    )
    state = torch.load(cfg["model_path"], map_location="cpu", weights_only=False)
    vq.load_state_dict(state.get("model_state_dict", state))
    vq.eval()
    decoders = {}
    vec_dir = Path(cfg["style_vectors_dir"])
    for s in ["pop", "jazz", "classical", "folk"]:
        dec_path = vec_dir / f"decoder_{s}.pt"
        if dec_path.exists():
            dec = Decoder(out_channels=cfg["in_channels"], embedding_dim=64)
            dec.load_state_dict(
                torch.load(str(dec_path), map_location="cpu", weights_only=False))
            dec.eval()
            decoders[s] = dec
    return vq, decoders, cfg


@st.cache_data(max_entries=40)
def run_style_transfer(midi_bytes: bytes, style: str,
                       emotion: str = "neutral") -> bytes:
    vq, decoders, cfg = load_vqvae()
    if style not in decoders:
        return b""
    pitch_low  = cfg["pitch_low"]
    pitch_high = cfg["pitch_high"]
    frame_rate = cfg["frame_rate"]

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        f.write(midi_bytes)
        tmp = f.name
    try:
        pm = pretty_midi.PrettyMIDI(tmp)
    finally:
        os.unlink(tmp)

    # Bug-04 根因修复：空 MIDI 直接返回空字节，UI 已在上游判 pred_n==0 拦截，
    # 这里是 src/style_transfer 之外的第二道防线。
    if sum(len(inst.notes) for inst in pm.instruments) == 0:
        return b""

    roll128 = pm.get_piano_roll(fs=frame_rate)
    roll48  = (roll128[pitch_low:pitch_high] > 0).astype(np.float32)
    T = roll48.shape[1]
    if T < 32:
        roll48 = np.pad(roll48, ((0, 0), (0, 32 - T)))
        T = 32
    pad = (8 - T % 8) % 8
    if pad:
        roll48 = np.pad(roll48, ((0, 0), (0, pad)))

    x = torch.from_numpy(roll48).float().unsqueeze(0)
    with torch.no_grad():
        z_q, _ = vq.encode(x)
        recon = decoders[style](z_q)

    recon_prob = recon.squeeze(0).numpy()[:, :T]
    recon48 = np.where(recon_prob > 0.5,
                       (recon_prob * 127).clip(1, 127), 0).astype(np.float32)
    recon128 = np.zeros((128, T), dtype=np.float32)
    recon128[pitch_low:pitch_high] = recon48

    try:
        bpm = pm.estimate_tempo()
    except ValueError:
        bpm = 120.0
    frame_dur = 1.0 / frame_rate
    melody_only = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, name="melody")
    for pitch in range(128):
        active = recon128[pitch] > 0
        if not np.any(active):
            continue
        changes = np.diff(active.astype(int))
        starts = (np.concatenate([[0], np.where(changes == 1)[0] + 1])
                  if active[0] else np.where(changes == 1)[0] + 1)
        ends = (np.concatenate([np.where(changes == -1)[0] + 1, [T]])
                if active[-1] else np.where(changes == -1)[0] + 1)
        for s, e in zip(starts, ends):
            vel = max(1, min(127, int(np.mean(recon128[pitch, s:e]))))
            inst.notes.append(pretty_midi.Note(vel, pitch, s * frame_dur, e * frame_dur))
    melody_only.instruments.append(inst)

    from src.style_postprocess import stylize
    # 情感 A 方案：bpm 经 emotion 缩放后传给 stylize，节拍/和弦时长自动适配
    bpm_emo = float(bpm) * EMOTION_PARAMS[emotion]["bpm_scale"]
    out = stylize(melody_only, style, tempo=bpm_emo)
    _apply_emotion_velocity(out, emotion)
    return midi_to_bytes(out)


def get_note_rows(midi):
    rows = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            rows.append({
                "音名": pretty_midi.note_number_to_name(note.pitch),
                "MIDI": note.pitch,
                "起始(s)": round(note.start, 3),
                "结束(s)": round(note.end, 3),
                "时值(s)": round(note.end - note.start, 3),
            })
    rows.sort(key=lambda x: x["起始(s)"])
    return rows


import pandas as pd


# ══════════════════════════════════════════════════════════════════
# 主界面：侧边栏导航 + 多页面
# ══════════════════════════════════════════════════════════════════

st.title("🎵 HummingMusic · 哼唱量化与风格迁移")

PAGE_INPUT = "🎤 输入与推理"
PAGE_ROLL  = "🎹 钢琴卷帘"
PAGE_SCORE = "📊 模型得分"
PAGE_NOTES = "📋 音符列表"
PAGE_STYLE = "🎨 风格迁移"
ALL_PAGES = [PAGE_INPUT, PAGE_ROLL, PAGE_SCORE, PAGE_NOTES, PAGE_STYLE]


# ── 侧边栏 ──
with st.sidebar:
    # Fix #1: 顶部页面导航（5 页 radio）
    st.markdown("### 📍 功能切换")
    page = st.radio(
        "page",
        ALL_PAGES,
        index=ALL_PAGES.index(st.session_state.get("page", PAGE_INPUT)),
        label_visibility="collapsed",
    )
    st.session_state["page"] = page
    st.markdown("---")

    # Fix #2: 侧边栏分组 expander（输入源 / 推理参数 / 合成 / 重置）
    with st.expander("🎙️ 输入源", expanded=True):
        use_demo = st.toggle(
            "🌰 使用演示样本",
            value=False,
            help="跳过数据集，直接用内置 example.m4a（你的哼唱）",
        )
        split = st.selectbox(
            "数据集分片", ["TEST", "VALID", "TRAIN"], index=0, disabled=use_demo,
        )
        st.caption("现场录音 / 上传 优先于数据集")
        recorded_audio = st.audio_input("🎙️ 现场录音")
        uploaded_file = st.file_uploader(
            "📤 上传哼唱音频",
            type=["wav", "mp3", "flac", "ogg", "m4a", "txt", "bin"],
        )

    with st.expander("⚙️ 推理参数", expanded=False):
        correct_octave = st.checkbox("八度偏移自动修正", value=True)
        peak_distance = st.slider(
            "峰值最小间距（帧，1帧=10ms）", 5, 50, 20,
        )
        peak_height_sigma = st.slider(
            "峰值高度阈值 (均值 + σ × std)", 0.0, 3.0, 0.5, 0.1,
        )

    with st.expander("🎼 合成音色", expanded=False):
        instrument_name = st.selectbox("音色", [
            "Acoustic Grand Piano",
            "Violin",
            "Flute",
            "Acoustic Guitar (nylon)",
            "Choir Aahs",
            "Synth Lead",
        ], index=0)

    st.markdown("---")
    if st.button("🔄 清空推理结果", use_container_width=True):
        for k in ["confirmed_input_fp", "last_result", "last_meta",
                  "infer_fp", "ds_idx"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.caption(
        "BiLSTM-CRF v5 · 13080 训练样本\n"
        "+ VQ-VAE 风格迁移 (pop/jazz/classical/folk)"
    )


# ── 输入源指纹（所有页面共享） ──
# 录音先落盘成 data/recordings/rec_<hash>.wav，再以 UploadedFile 兼容对象走
# 与「上传文件」完全一致的 validate / extract_features 路径。
if recorded_audio is not None:
    recorded_file = _save_recording_to_disk(recorded_audio)
    audio_source = _AudioSource(
        recorded_file,
        label=f"录音已保存：`{recorded_file.path}`",
        mode_label="现场录音",
        key_name=Path(recorded_file.name).stem,
    )
elif uploaded_file is not None:
    audio_source = _AudioSource(
        uploaded_file,
        label=f"上传：{uploaded_file.name}",
        mode_label=f"上传 {uploaded_file.name}",
        key_name=Path(uploaded_file.name).stem,
    )
else:
    audio_source = None

if audio_source is not None:
    current_input_fp = (f"file:{audio_source.name}:"
                        f"{getattr(audio_source, 'size', 0)}")
elif use_demo:
    current_input_fp = "demo"
else:
    current_input_fp = None  # 数据集模式：slider 实时刷新


def _render_run_button(label_suffix: str = ""):
    """渲染"开始推理"按钮，未触发则 st.stop()。"""
    c1, c2 = st.columns([1, 4])
    with c1:
        clicked = st.button(
            f"🚀 开始推理{label_suffix}",
            type="primary",
            use_container_width=True,
            key=f"run_btn_{current_input_fp}",
        )
    if clicked:
        st.session_state["confirmed_input_fp"] = current_input_fp
    with c2:
        if st.session_state.get("confirmed_input_fp") == current_input_fp:
            st.success("✅ 已触发推理")
        else:
            st.info("👆 点击按钮开始推理（录视频时可手动触发，避免上传即出结果）")
    if st.session_state.get("confirmed_input_fp") != current_input_fp:
        st.stop()


# ══════════════════════════════════════════════════════════════════
# 输入预处理 + 推理（每次脚本运行都执行，但结果通过 session_state 缓存）
# ══════════════════════════════════════════════════════════════════

upload_feat: np.ndarray | None = None
upload_wav_bytes: bytes | None = None

if audio_source is not None:
    ok, err_msg, duration_sec, sr = validate_uploaded_audio(audio_source)
    if not ok:
        if "格式" in err_msg:
            st.error(f"❌ {err_msg}")
            st.info("💡 支持的格式：WAV、MP3、FLAC、OGG、M4A")
        else:
            st.warning(f"⚠️ {err_msg}")
            st.info("💡 建议：录制至少 2 秒以上的哼唱片段，以获得准确的量化结果。")
        st.stop()
    st.success(f"✅ {audio_source.label}（{duration_sec:.1f}s，{sr}Hz）")
    _render_run_button(" / 重新推理")
    suffix = Path(getattr(audio_source, "name", "recording.wav")).suffix or ".wav"
    try:
        with st.spinner("提取 CREPE 特征中（约 5–15s）..."):
            upload_feat, upload_wav_bytes = extract_features_from_bytes(
                audio_source.getbuffer().tobytes(), suffix,
            )
        st.success(f"✅ 特征提取完成：{upload_feat.shape[0]} 帧 "
                   f"({int((upload_feat[:, 1] > 0).sum())} 有效)")
    except LowEnergyError as e:
        # Bug-04 根因层：CREPE 前 RMS 能量门控触发，省去整条空推理链路
        st.error(f"❌ 音频能量过低（{e.rms_dbfs:.1f} dBFS < {e.threshold_dbfs:.0f} dBFS）")
        st.info("💡 建议：贴近麦克风、提高音量、安静环境下录制，时长 3–15s")
        st.stop()
    except Exception as e:
        st.error(f"❌ 特征提取失败：{e}")
        st.stop()

with st.spinner("加载模型..."):
    model, device = load_model()

upload_mode = upload_feat is not None
if upload_mode:
    st.info(f"🎤 自定义音频模式：{audio_source.mode_label}（跳过 GT 评估）")
    item = {
        "features": torch.from_numpy(upload_feat),
        "n_frames": len(upload_feat),
        "key":      audio_source.key_name,
    }
    ds_idx_for_fp = None
elif use_demo:
    st.info("🎤 演示模式：使用内置 example.m4a（你的哼唱），跳过 GT 评估")
    _render_run_button(" / 重新推理")
    item = load_demo_item()
    ds_idx_for_fp = None
else:
    # Fix #3: 数据集模式增加 «‹ / ›» 翻页按钮，比拖 slider 顺手
    _split_hint = {
        "TEST":  "769 样本，约 0.5s",
        "VALID": "765 样本，约 0.5s",
        "TRAIN": "13080 样本，首次加载约 8-12s（已缓存后秒返）",
    }[split]
    with st.spinner(f"加载 {split} 数据集（{_split_hint}）..."):
        _t0_ds = time.time()
        ds = load_dataset(split, correct_octave=correct_octave)
    st.success(f"已加载 {len(ds)} 条样本（耗时 {time.time() - _t0_ds:.1f}s）")

    col1, col2, col3, col4 = st.columns([3, 1.4, 0.6, 0.6])
    with col1:
        idx = st.slider(
            "样本索引",
            0, len(ds) - 1,
            min(st.session_state.get("ds_idx", 0), len(ds) - 1),
            key="ds_idx_slider",
        )
    with col2:
        key_input = st.text_input("或直接输入 key", label_visibility="visible")
    with col3:
        if st.button("‹", help="上一条", use_container_width=True):
            idx = max(0, idx - 1)
            st.session_state["ds_idx"] = idx
            st.rerun()
    with col4:
        if st.button("›", help="下一条", use_container_width=True):
            idx = min(len(ds) - 1, idx + 1)
            st.session_state["ds_idx"] = idx
            st.rerun()
    st.session_state["ds_idx"] = idx

    if key_input:
        key_list = [ds.keys[i] for i in range(len(ds))]
        if key_input in key_list:
            idx = key_list.index(key_input)
            st.info(f"找到 key: {key_input}，index={idx}")
        else:
            st.warning(f"未找到 key: {key_input}")
    item = ds[idx]
    ds_idx_for_fp = (split, idx, correct_octave)

key = item["key"]
n_frames = item["n_frames"]
duration = n_frames * FRAME_STEP

# Fix #4: 推理结果缓存（按 input + params 指纹），换页不重跑
infer_fp = (current_input_fp or f"ds:{ds_idx_for_fp}",
            peak_distance, peak_height_sigma)

if st.session_state.get("infer_fp") != infer_fp:
    with st.spinner("模型推理..."):
        result, elapsed = run_inference_with_timeout(
            item, model, device, peak_distance, peak_height_sigma,
        )
    if result is None:
        st.error(f"⏱️ 推理超时（{elapsed:.1f}s > {INFERENCE_TIMEOUT}s）")
        st.info("💡 建议：缩短音频时长（10–30s），或降低 CREPE 精度到 small")
        st.stop()
    st.session_state["last_result"] = result
    st.session_state["infer_fp"] = infer_fp
else:
    result = st.session_state["last_result"]

# 空音符兜底
_pred_n_check = sum(len(i.notes) for i in result["pred_midi"].instruments)
_valid_ratio = float(result["valid_mask"].mean()) if result["n"] else 0.0
if _pred_n_check == 0:
    st.warning("⚠️ 未检测到有效音符：哼唱信号可能过弱、为静音或噪声。")
    diag = []
    if _valid_ratio < 0.1:
        diag.append(f"有效帧占比仅 {_valid_ratio:.1%}（CREPE 置信度过低）")
    if len(result["peaks"]) == 0:
        diag.append("BiLSTM 未在任何帧检出音符起始")
    st.caption("诊断：" + ("；".join(diag) if diag else "模型输出全空"))
    st.info("💡 贴近麦克风、提高音量、安静环境下录制，时长 3-15s")
    st.stop()

no_gt = use_demo or upload_mode
if no_gt:
    gt_midi = None
    metrics = None
else:
    gt_midi_path = Path(MIDI_DIR) / f"{key}.mid"
    gt_midi = pretty_midi.PrettyMIDI(str(gt_midi_path))
    metrics = compute_note_metrics(result["pred_midi"], gt_midi)

if upload_mode:
    orig_wav_bytes = upload_wav_bytes
elif use_demo:
    orig_wav_bytes = DEMO_WAV.read_bytes()
else:
    p = Path(WAV_DIR) / f"{key}.wav"
    orig_wav_bytes = p.read_bytes() if p.exists() else None


# ══════════════════════════════════════════════════════════════════
# 当前样本面包屑（所有页面共享，置于内容上方）
# ══════════════════════════════════════════════════════════════════

st.markdown(
    f"📌 **当前样本：** `{key}`  |  ⏱️ **时长：** {duration:.1f}s  "
    f"|  🎞️ **帧数：** {n_frames}"
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# 渲染页面
# ══════════════════════════════════════════════════════════════════

if page == PAGE_INPUT:
    st.subheader(PAGE_INPUT)

    if no_gt:
        pred_n = sum(len(i.notes) for i in result["pred_midi"].instruments)
        c1, c2, c3 = st.columns(3)
        c1.metric("预测音符数", pred_n)
        c2.metric("时长", f"{duration:.1f}s")
        c3.metric("帧数", n_frames)
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Note Accuracy", f"{metrics.note_accuracy:.3f}")
        c2.metric("Precision",     f"{metrics.precision:.3f}")
        c3.metric("F1",            f"{metrics.f1:.3f}")
        c4.metric("GT 音符数",     metrics.n_gt)
        c5.metric("预测音符数",    metrics.n_pred)

    st.markdown("---")
    st.markdown("#### 🔊 音频对比")
    if no_gt:
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown("**原始哼唱**")
            if orig_wav_bytes:
                st.audio(orig_wav_bytes, format="audio/wav")
            else:
                st.warning("WAV 不存在")
        with ac2:
            st.markdown("**预测 MIDI 合成**")
            with st.spinner("合成预测..."):
                pred_wav = synthesize_midi(
                    midi_to_bytes(result["pred_midi"]), instrument_name)
            st.audio(pred_wav, format="audio/wav")
    else:
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            st.markdown("**原始哼唱**")
            if orig_wav_bytes:
                st.audio(orig_wav_bytes, format="audio/wav")
            else:
                st.warning("WAV 文件不存在")
        with ac2:
            st.markdown("**GT MIDI 合成**")
            with st.spinner("合成 GT..."):
                gt_wav = synthesize_midi(midi_to_bytes(gt_midi), instrument_name)
            st.audio(gt_wav, format="audio/wav")
        with ac3:
            st.markdown("**预测 MIDI 合成**")
            with st.spinner("合成预测..."):
                pred_wav = synthesize_midi(
                    midi_to_bytes(result["pred_midi"]), instrument_name)
            st.audio(pred_wav, format="audio/wav")

    st.caption("📍 想看可视化对比？左侧切到「🎹 钢琴卷帘」。"
               "想要风格迁移？切到「🎨 风格迁移」。")

elif page == PAGE_ROLL:
    st.subheader(PAGE_ROLL)
    if no_gt:
        st.caption("当前为自定义/演示输入，无 GT，只显示预测（蓝色边框）")
    else:
        st.caption("绿色填充 = GT，蓝色边框 = 预测")
    fig_roll = plot_piano_roll(result["pred_midi"], gt_midi, duration, title=f"{key}")
    if fig_roll:
        st.pyplot(fig_roll, use_container_width=True)
        plt.close(fig_roll)
    else:
        st.warning("无音符可绘制")

elif page == PAGE_SCORE:
    st.subheader(PAGE_SCORE)
    st.caption("BiLSTM 第二维（B 类）得分曲线 + CREPE 置信度")
    fig_score = plot_scores(result)
    st.pyplot(fig_score, use_container_width=True)
    plt.close(fig_score)

elif page == PAGE_NOTES:
    st.subheader(PAGE_NOTES)
    if no_gt:
        st.caption("当前为自定义/演示输入，无 GT，只展示预测音符")
        st.dataframe(
            pd.DataFrame(get_note_rows(result["pred_midi"])),
            use_container_width=True, height=520,
        )
    else:
        col_gt, col_pred = st.columns(2)
        with col_gt:
            st.markdown("**GT 音符**")
            st.dataframe(
                pd.DataFrame(get_note_rows(gt_midi)),
                use_container_width=True, height=520,
            )
        with col_pred:
            st.markdown("**预测音符**")
            st.dataframe(
                pd.DataFrame(get_note_rows(result["pred_midi"])),
                use_container_width=True, height=520,
            )

elif page == PAGE_STYLE:
    st.subheader(PAGE_STYLE)
    st.caption("VQ-VAE 风格独立 decoder + style_postprocess.stylize 多轨")

    # 情感选择：作用于 stylize 的 tempo 与输出 velocity
    EMO_OPTS = [
        ("😐 中性", "neutral"),
        ("😊 欢快", "happy"),
        ("😢 忧伤", "sad"),
        ("🍵 平静", "calm"),
        ("🔥 激昂", "intense"),
    ]
    emo_label = st.radio(
        "🎭 情感",
        [lbl for lbl, _ in EMO_OPTS],
        index=0, horizontal=True,
        help="作用于 bpm 与 velocity：例如 intense 拉快 1.18× 且整体更响、sad 放慢 0.85× 且更轻",
    )
    emotion = dict(EMO_OPTS)[emo_label]
    p = EMOTION_PARAMS[emotion]
    st.caption(
        f"`{emotion}` · bpm × {p['bpm_scale']:.2f}  ·  "
        f"velocity × {p['vel_scale']:.2f} + {p['vel_offset']:+d}  ·  "
        f"range × {p['vel_range']:.2f}"
    )

    with st.spinner("加载 VQ-VAE..."):
        load_vqvae()

    pred_midi_bytes = midi_to_bytes(result["pred_midi"])

    st.markdown("#### 🎼 迁移前（量化旋律）")
    with st.spinner("合成原始旋律..."):
        orig_styled = synthesize_midi(pred_midi_bytes, instrument_name)
    st.audio(orig_styled, format="audio/wav")

    st.markdown("---")
    st.markdown(f"#### 🎨 迁移后（四种风格 × `{emotion}`）")
    style_cols = st.columns(4)
    for col, style in zip(style_cols, ["pop", "jazz", "classical", "folk"]):
        with col:
            st.markdown(f"**{style.capitalize()}**")
            with st.spinner(f"{style} 风格化中..."):
                styled_bytes = run_style_transfer(pred_midi_bytes, style, emotion)
            if styled_bytes:
                styled_wav = synthesize_midi(styled_bytes, instrument_name)
                st.audio(styled_wav, format="audio/wav")
            else:
                st.warning(f"decoder_{style}.pt 不存在")
