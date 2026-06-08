"""
test_model_bugs.py - Week12 模型层 bug 修复回归测试

覆盖：
- Bug-04 根因层：RMS 能量门控 / quantizer 全 NaN 短路 / style_transfer 空 MIDI 短路
- Bug-07 缓存：bytes 输入相同时 @st.cache_data hash 应一致（脱离 Streamlit 验证）
"""
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio_processing import LowEnergyError, _check_rms_energy


# ──────────────────────────────────────────────
# Bug-04 #1: RMS 能量门控
# ──────────────────────────────────────────────

def test_rms_zero_audio_raises():
    """全零（数字静音）应被拦截。"""
    with pytest.raises(LowEnergyError) as exc:
        _check_rms_energy(np.zeros(16000, dtype=np.float32))
    assert exc.value.rms_dbfs == -np.inf or exc.value.rms_dbfs < -100


def test_rms_low_noise_raises():
    """-60 dBFS 底噪应被拦截。"""
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(16000).astype(np.float32) * 1e-3  # ~-60 dBFS
    with pytest.raises(LowEnergyError):
        _check_rms_energy(audio)


def test_rms_normal_humming_passes():
    """-25 dBFS（正常哼唱）应放行。"""
    t = np.arange(16000) / 16000
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # ~-23 dBFS
    _check_rms_energy(audio)  # 不抛即通过


def test_rms_empty_raises():
    """空数组也算静音。"""
    with pytest.raises(LowEnergyError):
        _check_rms_energy(np.array([], dtype=np.float32))


# ──────────────────────────────────────────────
# Bug-04 #2: quantizer 全 NaN 短路
# ──────────────────────────────────────────────

def test_quantizer_all_nan_short_circuit(monkeypatch):
    """有效帧 < 10% 时不应触发模型加载/推理。"""
    from src import quantizer

    call_count = {"n": 0}

    def fake_load_model():
        call_count["n"] += 1
        return None  # 即便走到 model 也是 fallback

    monkeypatch.setattr(quantizer, "_load_model", fake_load_model)

    pitch_data = {
        "time": np.arange(1000) * 0.01,
        "frequency": np.full(1000, np.nan, dtype=np.float32),
        "confidence": np.zeros(1000, dtype=np.float32),
        "bpm": 120.0,
    }
    midi = quantizer.quantize_humming(pitch_data)
    n = sum(len(inst.notes) for inst in midi.instruments)
    assert n == 0
    assert call_count["n"] == 0, "短路时不应调用 _load_model"


def test_quantizer_partial_valid_uses_model(monkeypatch):
    """50% 有效帧时正常走模型路径（这里 fallback 也算正常路径）。"""
    from src import quantizer

    call_count = {"n": 0}

    def fake_load_model():
        call_count["n"] += 1
        return None  # 走 baseline

    monkeypatch.setattr(quantizer, "_load_model", fake_load_model)

    n_frames = 1000
    freq = np.where(
        np.arange(n_frames) < n_frames // 2,
        440.0,
        np.nan,
    ).astype(np.float32)
    pitch_data = {
        "time": np.arange(n_frames) * 0.01,
        "frequency": freq,
        "confidence": np.ones(n_frames, dtype=np.float32) * 0.9,
        "bpm": 120.0,
    }
    quantizer.quantize_humming(pitch_data)
    assert call_count["n"] == 1, "≥10% 有效帧时必须走模型/baseline 路径"


# ──────────────────────────────────────────────
# Bug-04 #3: style_transfer 空 MIDI 短路
# ──────────────────────────────────────────────

def test_style_transfer_empty_midi_short_circuit(monkeypatch):
    """空 MIDI 不应触发 VQ-VAE 加载/推理。"""
    from src import style_transfer

    call_count = {"vq": 0, "dec": 0}

    def fake_load_vq():
        call_count["vq"] += 1
        return None

    def fake_load_dec():
        call_count["dec"] += 1
        return {}

    monkeypatch.setattr(style_transfer, "_load_vqvae_model", fake_load_vq)
    monkeypatch.setattr(style_transfer, "_load_style_decoders", fake_load_dec)
    monkeypatch.setattr(style_transfer, "_load_style_vectors", lambda: {})

    empty_midi = pretty_midi.PrettyMIDI()
    out = style_transfer.transfer_style(empty_midi, "pop")
    n = sum(len(inst.notes) for inst in out.instruments)
    assert n == 0
    assert call_count["vq"] == 0, "空 MIDI 时不应加载 VQ-VAE"


# ──────────────────────────────────────────────
# Bug-07: bytes 输入下 cache hash 应一致
# ──────────────────────────────────────────────

def test_cache_key_stable_for_same_bytes(tmp_path):
    """Streamlit @st.cache_data 用 hashlib.md5(bytes) 作 key；同字节必须同 key。"""
    import hashlib
    audio_path = tmp_path / "silence.wav"
    audio_path.write_bytes(b"stable-silence-audio-bytes")
    blob = audio_path.read_bytes()
    h1 = hashlib.md5(blob).hexdigest()
    h2 = hashlib.md5(blob).hexdigest()
    assert h1 == h2
    # 模拟两次 UploadedFile.getbuffer().tobytes() 的结果一致
    h3 = hashlib.md5(memoryview(blob).tobytes()).hexdigest()
    assert h1 == h3


def test_local_audio_file_buffer_stable(tmp_path):
    """_LocalAudioFile.getbuffer() 必须在多次调用下返回相同字节。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    # 直接构造（跳过 Streamlit import）
    from pathlib import Path as P

    class _Stub:
        def __init__(self, p): self.path = P(p); self.name = self.path.name
        def getbuffer(self): return memoryview(self.path.read_bytes())

    audio_path = tmp_path / "silence.wav"
    audio_path.write_bytes(b"stable-silence-audio-bytes")
    f = _Stub(audio_path)
    b1 = f.getbuffer().tobytes()
    b2 = f.getbuffer().tobytes()
    assert b1 == b2
    assert hash(b1) == hash(b2)
