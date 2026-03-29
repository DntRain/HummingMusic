"""
test_renderer.py - 渲染模块单元测试

覆盖：正常输入、边界输入（空MIDI）、fallback 渲染
"""

from pathlib import Path

import numpy as np
import pretty_midi
import pytest


def _create_test_midi(n_notes: int = 4) -> pretty_midi.PrettyMIDI:
    """创建测试用 MIDI 对象。"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0, name="melody")
    for i in range(n_notes):
        note = pretty_midi.Note(
            velocity=100,
            pitch=60 + i,
            start=i * 0.5,
            end=(i + 1) * 0.5,
        )
        instrument.notes.append(note)
    midi.instruments.append(instrument)
    return midi


class TestRenderAudio:
    """render_audio 接口测试。"""

    def test_normal_render(self):
        """正常 MIDI 应成功渲染为 WAV。"""
        from src.renderer import render_audio

        midi = _create_test_midi()
        wav_path = render_audio(midi)

        assert wav_path.endswith(".wav")
        assert Path(wav_path).exists()
        assert Path(wav_path).stat().st_size > 0

        # 清理
        Path(wav_path).parent.rmdir() if not any(
            Path(wav_path).parent.iterdir()
        ) else None

    def test_output_in_tmp_dir(self):
        """输出应在 tmp/ 目录下的 UUID 子目录中。"""
        from src.renderer import render_audio

        midi = _create_test_midi()
        wav_path = render_audio(midi)

        assert "tmp" in wav_path
        # UUID 子目录应存在
        parent = Path(wav_path).parent
        assert parent.exists()
        # 应同时生成 MIDI 文件
        midi_path = parent / "output.mid"
        assert midi_path.exists()

    def test_empty_midi(self):
        """空 MIDI（无音符）应能渲染（静音文件）。"""
        from src.renderer import render_audio

        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        midi.instruments.append(
            pretty_midi.Instrument(program=0, name="melody")
        )

        wav_path = render_audio(midi)
        assert Path(wav_path).exists()

    def test_single_note(self):
        """单音符 MIDI 应能渲染。"""
        from src.renderer import render_audio

        midi = _create_test_midi(n_notes=1)
        wav_path = render_audio(midi)
        assert Path(wav_path).exists()
        assert Path(wav_path).stat().st_size > 0


class TestFallbackRender:
    """pretty_midi fallback 渲染测试。"""

    def test_pretty_midi_render(self):
        """使用 pretty_midi 内置合成应生成有效 WAV。"""
        from src.renderer import _render_with_pretty_midi

        midi = _create_test_midi()
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        success = _render_with_pretty_midi(midi, tmp.name)
        assert success
        assert Path(tmp.name).exists()
        assert Path(tmp.name).stat().st_size > 0

        Path(tmp.name).unlink(missing_ok=True)


class TestTmpDirManagement:
    """临时目录管理测试。"""

    def test_unique_directories(self):
        """每次调用应创建不同的 UUID 目录。"""
        from src.renderer import _ensure_tmp_dir

        dir1 = _ensure_tmp_dir()
        dir2 = _ensure_tmp_dir()
        assert dir1 != dir2
        assert dir1.exists()
        assert dir2.exists()

        # 清理
        dir1.rmdir()
        dir2.rmdir()
