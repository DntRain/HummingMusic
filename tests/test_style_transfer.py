"""
test_style_transfer.py - 风格迁移模块单元测试

覆盖：正常输入、边界输入（空MIDI、单音符MIDI）、异常输入（无效风格标签）
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


def _create_test_midi(n_notes: int = 4, bpm: float = 120.0) -> pretty_midi.PrettyMIDI:
    """创建测试用 MIDI 对象。"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0, name="melody")
    for i in range(n_notes):
        note = pretty_midi.Note(
            velocity=100,
            pitch=60 + i * 2,
            start=i * 0.5,
            end=(i + 1) * 0.5,
        )
        instrument.notes.append(note)
    midi.instruments.append(instrument)
    return midi


class TestPianoRollConversion:
    """Piano roll 编解码测试。"""

    def test_midi_to_piano_roll(self):
        """正常MIDI应生成合理的 piano roll。"""
        from src.style_transfer import midi_to_piano_roll

        midi = _create_test_midi()
        roll = midi_to_piano_roll(midi, fs=100)
        assert roll.shape[0] == 128
        assert roll.shape[1] > 0
        assert np.any(roll > 0)

    def test_piano_roll_to_midi(self):
        """Piano roll 转回 MIDI 应保留音符。"""
        from src.style_transfer import piano_roll_to_midi

        roll = np.zeros((128, 100), dtype=np.float32)
        roll[60, 10:30] = 100  # C4 from frame 10 to 30
        roll[64, 40:60] = 100  # E4 from frame 40 to 60

        midi = piano_roll_to_midi(roll, bpm=120.0)
        assert isinstance(midi, pretty_midi.PrettyMIDI)
        assert len(midi.instruments) > 0
        assert len(midi.instruments[0].notes) >= 2

    def test_empty_piano_roll(self):
        """空 piano roll 应生成无音符 MIDI。"""
        from src.style_transfer import piano_roll_to_midi

        roll = np.zeros((128, 50), dtype=np.float32)
        midi = piano_roll_to_midi(roll, bpm=120.0)
        assert isinstance(midi, pretty_midi.PrettyMIDI)
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        assert total_notes == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch 未安装，CI 中跳过")
class TestVQVAE:
    """VQ-VAE 模型测试（需要真实 torch）。"""

    def test_vector_quantizer(self):
        """向量量化器应返回正确形状。"""
        import torch

        from src.style_transfer import VectorQuantizer

        vq = VectorQuantizer(codebook_size=64, embedding_dim=32)
        z = torch.randn(2, 32, 10)
        z_q, indices, loss = vq(z)
        assert z_q.shape == z.shape
        assert indices.shape == (2, 10)
        assert loss.item() >= 0

    def test_encoder_decoder(self):
        """编码器和解码器维度应匹配。"""
        import torch

        from src.style_transfer import Decoder, Encoder

        enc = Encoder(in_channels=128, channels=[64, 128, 256], embedding_dim=64)
        dec = Decoder(out_channels=128, channels=[256, 128, 64], embedding_dim=64)

        x = torch.randn(1, 128, 64)
        z = enc(x)
        recon = dec(z)
        assert recon.shape[0] == 1
        assert recon.shape[1] == 128

    def test_style_vqvae_forward(self):
        """完整 VQ-VAE 前向传播应返回正确形状。"""
        import torch

        from src.style_transfer import StyleVQVAE

        model = StyleVQVAE(
            in_channels=128, codebook_size=64, embedding_dim=32
        )
        model.eval()

        x = torch.randn(1, 128, 64)
        with torch.no_grad():
            recon, indices, vq_loss = model(x)

        assert recon.shape[1] == 128
        assert vq_loss.item() >= 0


class TestTransferStyle:
    """transfer_style 接口测试。"""

    def test_invalid_style(self):
        """无效风格标签应抛出 ValueError。"""
        from src.style_transfer import transfer_style

        midi = _create_test_midi()
        with pytest.raises(ValueError, match="无效的风格"):
            transfer_style(midi, "rock")

    def test_fallback_pop(self):
        """fallback 模式下 pop 风格应返回有效 MIDI。"""
        from src.style_transfer import transfer_style

        midi = _create_test_midi()
        result = transfer_style(midi, "pop")
        assert isinstance(result, pretty_midi.PrettyMIDI)
        assert len(result.instruments) > 0

    def test_fallback_jazz(self):
        """fallback 模式下 jazz 风格应返回有效 MIDI。"""
        from src.style_transfer import transfer_style

        midi = _create_test_midi()
        result = transfer_style(midi, "jazz")
        assert isinstance(result, pretty_midi.PrettyMIDI)

    def test_fallback_classical(self):
        """fallback 模式下 classical 风格应返回有效 MIDI。"""
        from src.style_transfer import transfer_style

        midi = _create_test_midi()
        result = transfer_style(midi, "classical")
        assert isinstance(result, pretty_midi.PrettyMIDI)

    def test_fallback_folk(self):
        """fallback 模式下 folk 风格应返回有效 MIDI。"""
        from src.style_transfer import transfer_style

        midi = _create_test_midi()
        result = transfer_style(midi, "folk")
        assert isinstance(result, pretty_midi.PrettyMIDI)

    def test_empty_midi(self):
        """空 MIDI（无音符）应不崩溃。"""
        from src.style_transfer import transfer_style

        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        midi.instruments.append(
            pretty_midi.Instrument(program=0, name="melody")
        )
        result = transfer_style(midi, "pop")
        assert isinstance(result, pretty_midi.PrettyMIDI)

    def test_single_note_midi(self):
        """单音符 MIDI 应正常处理。"""
        from src.style_transfer import transfer_style

        midi = _create_test_midi(n_notes=1)
        result = transfer_style(midi, "jazz")
        assert isinstance(result, pretty_midi.PrettyMIDI)
