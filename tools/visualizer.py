"""
tools/visualizer.py - BiLSTM-CRF 量化器交互式可视化工具

用法：
    streamlit run tools/visualizer.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import json
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
from train.metrics import compute_note_metrics

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
DATA_ROOT   = "/run/media/DontRain/DATA_NANO/HumTrans"
FEAT_DIR    = "data/features_crepe"
CKPT_PATH   = "models/quantizer_v4/bilstm_crf.pt"
SPLIT_JSON  = f"{DATA_ROOT}/train_valid_test_keys.json"
WAV_DIR     = f"{DATA_ROOT}/all_wav/wav_data_sync_with_midi"
MIDI_DIR    = f"{DATA_ROOT}/midi_data"
FRAME_STEP  = 0.01  # 秒/帧
SF2_PATH    = str(Path(__file__).parent.parent.parent /
               "YOLO11n_Furnas/python312/lib/python3.12/site-packages/pretty_midi/TimGM6mb.sf2")
SYNTH_SR    = 22050


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
def load_dataset(split: str):
    import logging
    logging.disable(logging.WARNING)
    ds = HumTransDataset(
        split_json=SPLIT_JSON, split=split,
        wav_dir=WAV_DIR, midi_dir=MIDI_DIR,
        feat_dir=FEAT_DIR,
    )
    logging.disable(logging.NOTSET)
    return ds


# ──────────────────────────────────────────────
# MIDI 合成
# ──────────────────────────────────────────────
@st.cache_data(max_entries=20)
def synthesize_midi(midi_bytes: bytes, instrument_name: str = "Acoustic Grand Piano") -> bytes:
    """将 MIDI bytes 合成为 WAV bytes（用于 st.audio 播放）。"""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        f.write(midi_bytes)
        tmp_path = f.name
    try:
        pm = pretty_midi.PrettyMIDI(tmp_path)
        # 统一音色为钢琴
        program = pretty_midi.instrument_name_to_program(instrument_name)
        for inst in pm.instruments:
            if not inst.is_drum:
                inst.program = program
        audio = pm.fluidsynth(fs=SYNTH_SR, sf2_path=SF2_PATH)
        # 归一化防止截幅
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.9
        buf = io.BytesIO()
        sf.write(buf, audio, SYNTH_SR, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    finally:
        os.unlink(tmp_path)


def midi_to_bytes(pm: pretty_midi.PrettyMIDI) -> bytes:
    """PrettyMIDI 对象序列化为 bytes。"""
    import tempfile, os
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
        emissions = model.fc(lstm_out)  # (1, T, 3)

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

    # GT（绿色，半透明填充）
    for pitch, start, end in gt_notes:
        rect = mpatches.FancyBboxPatch(
            (start, pitch - 0.45), end - start, 0.9,
            boxstyle="round,pad=0.02",
            facecolor="#2ecc71", edgecolor="#27ae60", linewidth=0.8, alpha=0.55,
        )
        ax.add_patch(rect)

    # 预测（蓝色，实线边框）
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

    # Y 轴标注音名
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
    n = result["n"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")

    # ── B 类得分 + 峰值 ──
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    ax.plot(time_arr, b_scores, color="#e74c3c", linewidth=0.8, label="B 分数（音符起始）")
    ax.axhline(height_thr, color="yellow", linewidth=0.8, linestyle="--", alpha=0.7, label=f"阈值 {height_thr:.2f}")
    if len(peaks):
        ax.scatter(time_arr[peaks], b_scores[peaks], color="yellow", s=25, zorder=5, label=f"峰值 ({len(peaks)}个)")
    ax.set_ylabel("得分", color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", facecolor="#2a2a4a", labelcolor="white", fontsize=8)
    ax.set_title("模型 B 类得分（音符起始检测）", color="white", fontsize=10)
    ax.grid(alpha=0.15, color="white", linestyle="--")

    # ── CREPE 置信度 + MIDI 音高 ──
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    ax2.plot(time_arr, confidence, color="#3498db", linewidth=0.8, alpha=0.8, label="CREPE 置信度")
    ax2.set_ylabel("置信度", color="white", fontsize=9)
    ax2.set_xlabel("时间 (s)", color="white", fontsize=9)
    ax2.tick_params(colors="white")
    ax2.legend(loc="upper right", facecolor="#2a2a4a", labelcolor="white", fontsize=8)
    ax2.grid(alpha=0.15, color="white", linestyle="--")

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# 主界面
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="HummingMusic 可视化",
    page_icon="🎵",
    layout="wide",
)
st.title("🎵 BiLSTM-CRF 量化器可视化")

# ── 侧边栏 ──
with st.sidebar:
    st.header("设置")
    split = st.selectbox("数据集分片", ["TEST", "VALID", "TRAIN"], index=0)
    peak_distance = st.slider("峰值最小间距（帧，1帧=10ms）", 5, 50, 20)
    peak_height_sigma = st.slider("峰值高度阈值 (均值 + σ * std)", 0.0, 3.0, 0.5, 0.1)
    st.markdown("---")
    instrument_name = st.selectbox("合成音色", [
        "Acoustic Grand Piano",
        "Violin",
        "Flute",
        "Acoustic Guitar (nylon)",
        "Choir Aahs",
        "Synth Lead",
    ], index=0)
    st.markdown("---")
    st.caption("模型: BiLSTM-CRF v4\nCREPE 特征 + GT 对齐\n全量 13080 训练样本")

# ── 加载资源 ──
with st.spinner("加载模型..."):
    model, device = load_model()

with st.spinner(f"加载 {split} 数据集..."):
    ds = load_dataset(split)

st.success(f"已加载 {len(ds)} 条样本")

# ── 样本选择 ──
col1, col2 = st.columns([3, 1])
with col1:
    idx = st.slider("样本索引", 0, len(ds) - 1, 0)
with col2:
    key_input = st.text_input("或直接输入 key")

if key_input:
    key_list = [ds.keys[i] for i in range(len(ds))]
    if key_input in key_list:
        idx = key_list.index(key_input)
        st.info(f"找到 key: {key_input}，index={idx}")
    else:
        st.warning(f"未找到 key: {key_input}")

item = ds[idx]
key = item["key"]
n_frames = item["n_frames"]
duration = n_frames * FRAME_STEP

st.markdown(f"**Key:** `{key}`  |  **时长:** {duration:.1f}s  |  **帧数:** {n_frames}")

# ── 推理 ──
with st.spinner("模型推理..."):
    result = run_inference(item, model, device, peak_distance, peak_height_sigma)

gt_midi_path = Path(MIDI_DIR) / f"{key}.mid"
gt_midi = pretty_midi.PrettyMIDI(str(gt_midi_path))
metrics = compute_note_metrics(result["pred_midi"], gt_midi)

# ── 指标卡片 ──
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Note Accuracy", f"{metrics.note_accuracy:.3f}")
c2.metric("Precision",     f"{metrics.precision:.3f}")
c3.metric("F1",            f"{metrics.f1:.3f}")
c4.metric("GT 音符数",     metrics.n_gt)
c5.metric("预测音符数",    metrics.n_pred)

# ── 音频播放 ──
st.subheader("音频对比")
audio_col1, audio_col2, audio_col3 = st.columns(3)

wav_path = Path(WAV_DIR) / f"{key}.wav"
with audio_col1:
    st.markdown("**原始哼唱**")
    if wav_path.exists():
        with open(wav_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")
    else:
        st.warning("WAV 文件不存在")

with audio_col2:
    st.markdown("**GT MIDI 合成**")
    with st.spinner("合成 GT..."):
        gt_wav = synthesize_midi(midi_to_bytes(gt_midi), instrument_name)
    st.audio(gt_wav, format="audio/wav")

with audio_col3:
    st.markdown("**预测 MIDI 合成**")
    with st.spinner("合成预测..."):
        pred_wav = synthesize_midi(midi_to_bytes(result["pred_midi"]), instrument_name)
    st.audio(pred_wav, format="audio/wav")

# ── 钢琴卷帘 ──
st.subheader("钢琴卷帘对比（绿色填充=GT，蓝色边框=预测）")
fig_roll = plot_piano_roll(result["pred_midi"], gt_midi, duration, title=f"{key}")
if fig_roll:
    st.pyplot(fig_roll, use_container_width=True)
    plt.close(fig_roll)

# ── 得分图 ──
st.subheader("模型内部得分")
fig_score = plot_scores(result)
st.pyplot(fig_score, use_container_width=True)
plt.close(fig_score)

# ── 音符对比表 ──
st.subheader("音符列表对比")
col_gt, col_pred = st.columns(2)

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
with col_gt:
    st.markdown("**GT 音符**")
    gt_rows = get_note_rows(gt_midi)
    st.dataframe(pd.DataFrame(gt_rows), use_container_width=True, height=400)

with col_pred:
    st.markdown("**预测音符**")
    pred_rows = get_note_rows(result["pred_midi"])
    st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, height=400)
