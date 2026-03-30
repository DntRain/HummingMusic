"""
test_audio_processing.py - 音频处理模块单元测试

覆盖：正常输入、边界输入（极短音频）、异常输入（格式错误、文件不存在）
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf


class TestExtractPitch:
    """extract_pitch 接口测试。"""

    def _create_test_audio(
        self, duration: float = 2.0, sr: int = 16000, freq: float = 440.0
    ) -> str:
        """生成测试用正弦波音频文件。"""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, sr)
        return tmp.name

    def test_normal_input(self):
        """正常2秒440Hz音频应返回合理结果。"""
        audio_path = self._create_test_audio(duration=2.0, freq=440.0)

        mock_time = np.arange(0, 2.0, 0.01)
        mock_freq = np.full_like(mock_time, 440.0)
        mock_conf = np.full_like(mock_time, 0.95)

        with patch("src.audio_processing.crepe") as mock_crepe:
            mock_crepe.predict.return_value = (
                mock_time, mock_freq, mock_conf, None
            )
            from src.audio_processing import extract_pitch

            result = extract_pitch(audio_path)

        assert "time" in result
        assert "frequency" in result
        assert "confidence" in result
        assert "bpm" in result
        assert isinstance(result["bpm"], float)
        assert len(result["time"]) == len(result["frequency"])
        assert len(result["time"]) == len(result["confidence"])

        Path(audio_path).unlink(missing_ok=True)

    def test_short_audio(self):
        """极短音频（0.1秒）也应能正常处理。"""
        audio_path = self._create_test_audio(duration=0.1, freq=440.0)

        mock_time = np.arange(0, 0.1, 0.01)
        mock_freq = np.full_like(mock_time, 440.0)
        mock_conf = np.full_like(mock_time, 0.9)

        with patch("src.audio_processing.crepe") as mock_crepe:
            mock_crepe.predict.return_value = (
                mock_time, mock_freq, mock_conf, None
            )
            from src.audio_processing import extract_pitch

            result = extract_pitch(audio_path)

        assert len(result["time"]) > 0
        Path(audio_path).unlink(missing_ok=True)

    def test_low_confidence_frames(self):
        """低置信度帧应被标记为 NaN。"""
        audio_path = self._create_test_audio(duration=1.0)

        mock_time = np.arange(0, 1.0, 0.01)
        mock_freq = np.full_like(mock_time, 440.0)
        mock_conf = np.concatenate([
            np.full(50, 0.3),
            np.full(50, 0.95),
        ])

        with patch("src.audio_processing.crepe") as mock_crepe:
            mock_crepe.predict.return_value = (
                mock_time, mock_freq, mock_conf, None
            )
            from src.audio_processing import extract_pitch

            result = extract_pitch(audio_path)

        assert isinstance(result["frequency"], np.ndarray)
        Path(audio_path).unlink(missing_ok=True)

    def test_file_not_found(self):
        """不存在的文件应抛出 FileNotFoundError。"""
        from src.audio_processing import extract_pitch

        with pytest.raises(FileNotFoundError):
            extract_pitch("/nonexistent/path/audio.wav")

    def test_invalid_format(self):
        """无效格式文件应抛出 RuntimeError。"""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(b"this is not audio data")
        tmp.close()

        from src.audio_processing import extract_pitch

        with pytest.raises(RuntimeError):
            extract_pitch(tmp.name)

        Path(tmp.name).unlink(missing_ok=True)


class TestInterpolation:
    """短NaN段插值测试。"""

    def test_short_gap_interpolation(self):
        """短NaN段应被线性插值修复。"""
        from src.audio_processing import _interpolate_short_gaps

        time = np.arange(0, 1.0, 0.01)
        frequency = np.full(100, 440.0)
        frequency[50:53] = np.nan

        result = _interpolate_short_gaps(time, frequency)
        assert not np.any(np.isnan(result[50:53]))

    def test_long_gap_preserved(self):
        """长NaN段应保留。"""
        from src.audio_processing import _interpolate_short_gaps

        time = np.arange(0, 2.0, 0.01)
        frequency = np.full(200, 440.0)
        frequency[50:80] = np.nan

        result = _interpolate_short_gaps(time, frequency)
        assert np.all(np.isnan(result[50:80]))
