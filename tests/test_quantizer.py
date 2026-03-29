"""
test_quantizer.py - 容错量化模块单元测试

覆盖：正常输入、边界输入（单音符、极短序列）、异常输入（全NaN频率）
"""

import numpy as np
import pretty_midi
import pytest
import torch


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

        pitch_data = {
            "time": np.arange(0, 1.0, 0.01),
            "frequency": np.full(100, 440.0),
            "confidence": np.full(100, 0.95),
            "bpm": 120.0,
        }
        features = _prepare_features(pitch_data)
        assert features.shape == (100, 4)
        assert features.dtype == np.float32


class TestBiLSTMCRF:
    """BiLSTM-CRF 模型测试。"""

    def test_model_forward(self):
        """模型前向推理应返回正确长度的标签序列。"""
        from src.quantizer import BiLSTMCRF

        model = BiLSTMCRF(input_dim=4, hidden_size=32, num_layers=1)
        model.eval()

        x = torch.randn(1, 50, 4)
        with torch.no_grad():
            tags = model(x)

        assert len(tags) == 1
        assert len(tags[0]) == 50
        # 所有标签应在 {0, 1, 2} 范围内
        assert all(t in {0, 1, 2} for t in tags[0])

    def test_model_batch(self):
        """批量推理应返回对应数量的结果。"""
        from src.quantizer import BiLSTMCRF

        model = BiLSTMCRF(input_dim=4, hidden_size=32, num_layers=1)
        model.eval()

        x = torch.randn(3, 30, 4)
        with torch.no_grad():
            tags = model(x)

        assert len(tags) == 3


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


class TestBaselineQuantize:
    """Fallback 基线量化测试。"""

    def test_normal_input(self):
        """正常输入应返回有效 PrettyMIDI。"""
        from src.quantizer import _baseline_quantize

        pitch_data = {
            "time": np.arange(0, 1.0, 0.01),
            "frequency": np.full(100, 440.0),
            "confidence": np.full(100, 0.95),
            "bpm": 120.0,
        }

        result = _baseline_quantize(pitch_data)
        assert isinstance(result, pretty_midi.PrettyMIDI)
        assert len(result.instruments) > 0

    def test_all_nan_frequency(self):
        """全NaN频率应返回空MIDI（无音符）。"""
        from src.quantizer import _baseline_quantize

        pitch_data = {
            "time": np.arange(0, 1.0, 0.01),
            "frequency": np.full(100, np.nan),
            "confidence": np.full(100, 0.0),
            "bpm": 120.0,
        }

        result = _baseline_quantize(pitch_data)
        assert isinstance(result, pretty_midi.PrettyMIDI)
        total_notes = sum(len(inst.notes) for inst in result.instruments)
        assert total_notes == 0

    def test_single_frame(self):
        """单帧输入应不崩溃。"""
        from src.quantizer import _baseline_quantize

        pitch_data = {
            "time": np.array([0.0]),
            "frequency": np.array([440.0]),
            "confidence": np.array([0.95]),
            "bpm": 120.0,
        }

        result = _baseline_quantize(pitch_data)
        assert isinstance(result, pretty_midi.PrettyMIDI)


class TestQuantizeHumming:
    """quantize_humming 接口测试（使用 fallback）。"""

    def test_fallback_mode(self):
        """模型未加载时应自动使用 fallback。"""
        from src.quantizer import quantize_humming

        pitch_data = {
            "time": np.arange(0, 2.0, 0.01),
            "frequency": np.full(200, 440.0),
            "confidence": np.full(200, 0.95),
            "bpm": 120.0,
        }

        result = quantize_humming(pitch_data)
        assert isinstance(result, pretty_midi.PrettyMIDI)
        assert len(result.instruments) > 0
