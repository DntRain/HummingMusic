"""
test_quantizer.py - 容错量化模块单元测试

覆盖：
- 预处理函数（F0→MIDI转换、节拍位置、特征提取）
- BiLSTM-CRF 模型（需要真实 torch）
- BIO 后处理
- RoundingBaselineQuantizer（正式 baseline 类）
  - 正常哼唱输入（含置信度波动）
  - 全静音输入
  - 单音输入
  - 含长静音段的输入
  - 极端音高（MIDI 0 和 127 附近）
- quantize_humming 接口（fallback 模式）
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pretty_midi
import pytest

# 判断 torch 是否真实可用（非 mock）
TORCH_AVAILABLE = (
    "torch" in sys.modules and not isinstance(sys.modules["torch"], MagicMock)
)


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _make_pitch_data(
    duration: float = 2.0,
    freq: float = 440.0,
    confidence: float = 0.95,
    bpm: float = 120.0,
    step: float = 0.01,
) -> dict:
    """构造标准 pitch_data 字典。"""
    time = np.arange(0, duration, step)
    n = len(time)
    return {
        "time": time,
        "frequency": np.full(n, freq),
        "confidence": np.full(n, confidence),
        "bpm": bpm,
    }


# ──────────────────────────────────────────────
# 预处理函数测试
# ──────────────────────────────────────────────

class TestPreprocessing:
    """预处理函数测试。"""

    def test_freq_to_midi_a4(self):
        """440Hz 应转换为 MIDI 69 (A4)。"""
        from src.quantizer import _freq_to_midi

        freq = np.array([440.0])
        result = _freq_to_midi(freq)
        assert np.isclose(result[0], 69.0)

    def test_freq_to_midi_nan(self):
        """NaN 频率应保持 NaN。"""
        from src.quantizer import _freq_to_midi

        freq = np.array([np.nan, 440.0, np.nan])
        result = _freq_to_midi(freq)
        assert np.isnan(result[0])
        assert np.isclose(result[1], 69.0)
        assert np.isnan(result[2])

    def test_freq_to_midi_c4(self):
        """261.63Hz 应约为 MIDI 60 (C4)。"""
        from src.quantizer import _freq_to_midi

        freq = np.array([261.63])
        result = _freq_to_midi(freq)
        assert abs(result[0] - 60.0) < 0.1

    def test_compute_beat_positions(self):
        """节拍位置应在 [0, 1) 范围内。"""
        from src.quantizer import _compute_beat_positions

        time = np.arange(0, 4.0, 0.01)
        bpm = 120.0
        result = _compute_beat_positions(time, bpm)
        assert np.all(result >= 0)
        assert np.all(result < 1)

    def test_prepare_features_shape(self):
        """特征矩阵形状应为 (N, 4)。"""
        from src.quantizer import _prepare_features

        pitch_data = _make_pitch_data(duration=1.0)
        features = _prepare_features(pitch_data)
        assert features.shape == (100, 4)
        assert features.dtype == np.float32


# ──────────────────────────────────────────────
# BiLSTM-CRF 模型测试
# ──────────────────────────────────────────────

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch 未安装，CI 中跳过")
class TestBiLSTMCRF:
    """BiLSTM-CRF 模型测试（需要真实 torch）。"""

    def test_model_forward(self):
        """模型前向推理应返回正确长度的标签序列。"""
        import torch

        from src.quantizer import BiLSTMCRF

        model = BiLSTMCRF(input_dim=4, hidden_size=32, num_layers=1)
        model.eval()

        x = torch.randn(1, 50, 4)
        with torch.no_grad():
            tags = model(x)

        assert len(tags) == 1
        assert len(tags[0]) == 50
        assert all(t in {0, 1, 2} for t in tags[0])

    def test_model_batch(self):
        """批量推理应返回对应数量的结果。"""
        import torch

        from src.quantizer import BiLSTMCRF

        model = BiLSTMCRF(input_dim=4, hidden_size=32, num_layers=1)
        model.eval()

        x = torch.randn(3, 30, 4)
        with torch.no_grad():
            tags = model(x)

        assert len(tags) == 3


# ──────────────────────────────────────────────
# BIO 后处理测试
# ──────────────────────────────────────────────

class TestBIOPostProcessing:
    """BIO 标注后处理测试。"""

    def test_single_note(self):
        """单个 B-I-I-I 序列应生成一个音符。"""
        from src.quantizer import _bio_to_notes

        tags = [1, 2, 2, 2, 0]  # B I I I O
        time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        midi_notes = np.array([69.0, 69.0, 69.0, 69.0, np.nan])

        notes = _bio_to_notes(tags, time, midi_notes)
        assert len(notes) == 1
        assert notes[0]["pitch"] == 69
        assert notes[0]["start"] == 0.0
        assert notes[0]["end"] == 0.4

    def test_multiple_notes(self):
        """多个 B-I 段应生成多个音符。"""
        from src.quantizer import _bio_to_notes

        tags = [1, 2, 0, 1, 2]  # B I O B I
        time = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        midi_notes = np.array([60.0, 60.0, np.nan, 72.0, 72.0])

        notes = _bio_to_notes(tags, time, midi_notes)
        assert len(notes) == 2
        assert notes[0]["pitch"] == 60
        assert notes[1]["pitch"] == 72

    def test_empty_tags(self):
        """全 O 标签应生成空列表。"""
        from src.quantizer import _bio_to_notes

        tags = [0, 0, 0]
        time = np.array([0.0, 0.1, 0.2])
        midi_notes = np.array([np.nan, np.nan, np.nan])

        notes = _bio_to_notes(tags, time, midi_notes)
        assert len(notes) == 0


# ──────────────────────────────────────────────
# RoundingBaselineQuantizer 测试
# ──────────────────────────────────────────────

class TestRoundingBaselineQuantizer:
    """RoundingBaselineQuantizer 正式 baseline 类测试。"""

    def _get_quantizer(self):
        from src.quantizer import RoundingBaselineQuantizer
        return RoundingBaselineQuantizer()

    # --- 正常哼唱输入（含置信度波动） ---

    def test_normal_humming(self):
        """正常哼唱：稳定 440Hz 高置信度应输出单个 A4 音符。"""
        q = self._get_quantizer()
        pitch_data = _make_pitch_data(duration=2.0, freq=440.0, confidence=0.95)
        result = q(pitch_data)

        assert isinstance(result, pretty_midi.PrettyMIDI)
        assert len(result.instruments) == 1
        notes = result.instruments[0].notes
        assert len(notes) == 1
        assert notes[0].pitch == 69  # A4
        assert notes[0].velocity == 80

    def test_confidence_fluctuation(self):
        """置信度波动：部分帧低置信度应被过滤为静音。"""
        q = self._get_quantizer()
        pitch_data = _make_pitch_data(duration=1.0, freq=440.0)
        # 中间 5 帧置信度降到 0.5（< 0.8 阈值）→ 产生短静音
        # 短于 0.2s 的静音段会被插值填充回来
        pitch_data["confidence"][45:50] = 0.5

        result = q(pitch_data)
        notes = result.instruments[0].notes
        # 短静音被插值填充，应仍合并为单个音符
        assert len(notes) == 1
        assert notes[0].pitch == 69

    def test_confidence_long_gap(self):
        """长置信度低谷：超过 0.2s 的低置信度段应分割为两个音符。"""
        q = self._get_quantizer()
        pitch_data = _make_pitch_data(duration=2.0, freq=440.0)
        # 0.5s ~ 0.8s (30帧=0.3s > 0.2s) 置信度很低
        pitch_data["confidence"][50:80] = 0.3

        result = q(pitch_data)
        notes = result.instruments[0].notes
        # 长静音段不被插值，应分裂为两个音符
        assert len(notes) == 2
        assert notes[0].pitch == 69
        assert notes[1].pitch == 69

    def test_two_distinct_pitches(self):
        """两段不同音高应输出两个不同音符。"""
        q = self._get_quantizer()
        pitch_data = _make_pitch_data(duration=2.0, freq=440.0)
        # 后半段改为 C5 (523.25 Hz → MIDI 72)
        pitch_data["frequency"][100:] = 523.25

        result = q(pitch_data)
        notes = result.instruments[0].notes
        assert len(notes) == 2
        assert notes[0].pitch == 69  # A4
        assert notes[1].pitch == 72  # C5

    # --- 全静音输入 ---

    def test_all_silence_nan(self):
        """全 NaN 频率应返回无音符的 MIDI。"""
        q = self._get_quantizer()
        pitch_data = _make_pitch_data(duration=1.0)
        pitch_data["frequency"][:] = np.nan

        result = q(pitch_data)
        assert isinstance(result, pretty_midi.PrettyMIDI)
        total_notes = sum(len(inst.notes) for inst in result.instruments)
        assert total_notes == 0

    def test_all_low_confidence(self):
        """全低置信度应等效于全静音。"""
        q = self._get_quantizer()
        pitch_data = _make_pitch_data(duration=1.0, freq=440.0, confidence=0.3)

        result = q(pitch_data)
        total_notes = sum(len(inst.notes) for inst in result.instruments)
        assert total_notes == 0

    # --- 单音输入 ---

    def test_single_frame(self):
        """单帧输入应不崩溃，生成一个极短音符。"""
        q = self._get_quantizer()
        pitch_data = {
            "time": np.array([0.0]),
            "frequency": np.array([440.0]),
            "confidence": np.array([0.95]),
            "bpm": 120.0,
        }

        result = q(pitch_data)
        assert isinstance(result, pretty_midi.PrettyMIDI)
        notes = result.instruments[0].notes
        assert len(notes) == 1
        assert notes[0].pitch == 69

    def test_two_frames_same_pitch(self):
        """两帧同一音高应合并为一个音符。"""
        q = self._get_quantizer()
        pitch_data = {
            "time": np.array([0.0, 0.01]),
            "frequency": np.array([440.0, 440.0]),
            "confidence": np.array([0.95, 0.95]),
            "bpm": 120.0,
        }

        result = q(pitch_data)
        notes = result.instruments[0].notes
        assert len(notes) == 1

    # --- 含长静音段的输入 ---

    def test_long_silence_in_middle(self):
        """中间有长静音段应分割为前后两个音符。"""
        q = self._get_quantizer()
        time = np.arange(0, 3.0, 0.01)  # 300 帧
        n = len(time)
        frequency = np.full(n, 440.0)
        confidence = np.full(n, 0.95)

        # 1.0s ~ 2.0s (100帧=1.0s >> 0.2s) 设为静音
        frequency[100:200] = np.nan

        pitch_data = {
            "time": time,
            "frequency": frequency,
            "confidence": confidence,
            "bpm": 120.0,
        }

        result = q(pitch_data)
        notes = result.instruments[0].notes
        assert len(notes) == 2
        # 第一段：0~1.0s, 第二段：2.0s~3.0s
        assert notes[0].end <= 1.01
        assert notes[1].start >= 1.99

    # --- 极端音高 ---

    def test_extreme_low_pitch(self):
        """极低频（~8.18Hz → MIDI 0）应被钳制到 MIDI 0。"""
        q = self._get_quantizer()
        # MIDI 0 = 8.1758 Hz
        pitch_data = _make_pitch_data(duration=0.5, freq=8.18)

        result = q(pitch_data)
        notes = result.instruments[0].notes
        assert len(notes) >= 1
        assert notes[0].pitch == 0

    def test_extreme_high_pitch(self):
        """极高频（~12544Hz → MIDI 127）应被钳制到 MIDI 127。"""
        q = self._get_quantizer()
        # MIDI 127 ≈ 12543.85 Hz
        pitch_data = _make_pitch_data(duration=0.5, freq=12544.0)

        result = q(pitch_data)
        notes = result.instruments[0].notes
        assert len(notes) >= 1
        assert notes[0].pitch == 127

    def test_near_midi_boundary(self):
        """接近 MIDI 边界的频率应正确四舍五入。"""
        q = self._get_quantizer()
        # MIDI 69.6 对应 freq = 440 * 2^(0.6/12) ≈ 455.52 Hz
        # 应四舍五入到 70 (Bb4)
        pitch_data = _make_pitch_data(duration=0.5, freq=455.52)

        result = q(pitch_data)
        notes = result.instruments[0].notes
        assert notes[0].pitch == 70  # Bb4

    # --- 力度 ---

    def test_velocity_is_80(self):
        """输出音符力度应统一为配置值 80。"""
        q = self._get_quantizer()
        pitch_data = _make_pitch_data(duration=1.0)

        result = q(pitch_data)
        for note in result.instruments[0].notes:
            assert note.velocity == 80

    # --- 输出格式 ---

    def test_output_has_single_instrument(self):
        """输出应包含恰好一个旋律轨道。"""
        q = self._get_quantizer()
        pitch_data = _make_pitch_data()

        result = q(pitch_data)
        assert len(result.instruments) == 1
        assert result.instruments[0].name == "melody"

    def test_note_timing_monotonic(self):
        """输出音符的起止时间应单调递增。"""
        q = self._get_quantizer()
        pitch_data = _make_pitch_data(duration=2.0)
        # 制造两个音符
        pitch_data["frequency"][100:] = 523.25

        result = q(pitch_data)
        notes = result.instruments[0].notes
        for i in range(len(notes) - 1):
            assert notes[i].end <= notes[i + 1].start + 1e-6


# ──────────────────────────────────────────────
# 旧 _baseline_quantize 兼容性测试
# ──────────────────────────────────────────────

class TestBaselineQuantizeLegacy:
    """旧 _baseline_quantize 函数兼容性测试（确保未被破坏）。"""

    def test_normal_input(self):
        """正常输入应返回有效 PrettyMIDI。"""
        from src.quantizer import _baseline_quantize

        pitch_data = _make_pitch_data(duration=1.0)
        result = _baseline_quantize(pitch_data)
        assert isinstance(result, pretty_midi.PrettyMIDI)
        assert len(result.instruments) > 0

    def test_all_nan_frequency(self):
        """全NaN频率应返回空MIDI。"""
        from src.quantizer import _baseline_quantize

        pitch_data = _make_pitch_data(duration=1.0)
        pitch_data["frequency"][:] = np.nan

        result = _baseline_quantize(pitch_data)
        total_notes = sum(len(inst.notes) for inst in result.instruments)
        assert total_notes == 0


# ──────────────────────────────────────────────
# quantize_humming 接口测试
# ──────────────────────────────────────────────

class TestQuantizeHumming:
    """quantize_humming 接口测试（无模型权重时使用 RoundingBaselineQuantizer）。"""

    def test_fallback_mode(self):
        """模型未加载时应自动使用 RoundingBaselineQuantizer。"""
        from src.quantizer import quantize_humming

        pitch_data = _make_pitch_data(duration=2.0)
        result = quantize_humming(pitch_data)
        assert isinstance(result, pretty_midi.PrettyMIDI)
        assert len(result.instruments) > 0
        # 应使用 baseline 的力度 80（而非旧版 100）
        assert result.instruments[0].notes[0].velocity == 80

    def test_fallback_empty_input(self):
        """全静音输入走 fallback 不应崩溃。"""
        from src.quantizer import quantize_humming

        pitch_data = _make_pitch_data(duration=1.0)
        pitch_data["frequency"][:] = np.nan

        result = quantize_humming(pitch_data)
        assert isinstance(result, pretty_midi.PrettyMIDI)
