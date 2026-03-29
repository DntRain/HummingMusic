"""
interfaces.py - 模块间接口定义

定义 HummingMusic 系统四个核心处理阶段的接口协议。
所有模块间通信必须严格通过此处定义的接口进行。
"""

from typing import Literal, Protocol

import numpy as np
import pretty_midi


# ──────────────────────────────────────────────
# 类型别名
# ──────────────────────────────────────────────

Style = Literal["pop", "jazz", "classical", "folk"]

VALID_STYLES: list[str] = ["pop", "jazz", "classical", "folk"]


# ──────────────────────────────────────────────
# 接口协议定义
# ──────────────────────────────────────────────

class PitchExtractor(Protocol):
    """音高提取器接口协议。"""

    def __call__(self, audio_path: str) -> dict:
        """
        从音频文件中提取基频(F0)信息。

        Args:
            audio_path: 输入音频文件的绝对或相对路径，支持 WAV/MP3 格式。

        Returns:
            dict: 包含以下字段:
                - time (np.ndarray): 时间戳数组，单位秒，shape=(N,)
                - frequency (np.ndarray): 基频数组，单位Hz，shape=(N,)。
                  低置信度帧值为 NaN。
                - confidence (np.ndarray): 置信度数组，范围[0,1]，shape=(N,)
                - bpm (float): 估计的每分钟节拍数
        """
        ...


class HummingQuantizer(Protocol):
    """哼唱量化器接口协议。"""

    def __call__(self, pitch_data: dict) -> pretty_midi.PrettyMIDI:
        """
        将连续音高数据量化为离散MIDI音符序列。

        Args:
            pitch_data: extract_pitch 的输出字典，包含 time, frequency,
                        confidence, bpm 四个字段。

        Returns:
            pretty_midi.PrettyMIDI: 量化后的MIDI对象，包含单个乐器轨道，
                                     音符已对齐到节拍网格。
        """
        ...


class StyleTransferEngine(Protocol):
    """风格迁移引擎接口协议。"""

    def __call__(
        self, midi: pretty_midi.PrettyMIDI, style: Style
    ) -> pretty_midi.PrettyMIDI:
        """
        将输入MIDI的风格迁移到目标风格。

        Args:
            midi: 输入的 PrettyMIDI 对象（通常来自量化器输出）。
            style: 目标风格，取值为 'pop', 'jazz', 'classical', 'folk' 之一。

        Returns:
            pretty_midi.PrettyMIDI: 风格迁移后的MIDI对象，可能包含多个轨道
                                     （旋律+伴奏）。
        """
        ...


class AudioRenderer(Protocol):
    """音频渲染器接口协议。"""

    def __call__(self, midi: pretty_midi.PrettyMIDI) -> str:
        """
        将MIDI对象渲染为可播放的WAV音频文件。

        Args:
            midi: 待渲染的 PrettyMIDI 对象。

        Returns:
            str: 生成的WAV文件绝对路径，位于 tmp/{uuid}/ 目录下。
        """
        ...


# ──────────────────────────────────────────────
# 接口函数签名（供直接导入使用）
# ──────────────────────────────────────────────

def extract_pitch(audio_path: str) -> dict:
    """
    从音频文件中提取基频(F0)信息。

    Args:
        audio_path: 输入音频文件路径，支持 WAV/MP3 格式。

    Returns:
        dict: 包含以下字段:
            - time (np.ndarray): 时间戳数组，单位秒，shape=(N,)
            - frequency (np.ndarray): 基频数组，单位Hz，shape=(N,)。
              低置信度帧值为 NaN。
            - confidence (np.ndarray): 置信度数组，范围[0,1]，shape=(N,)
            - bpm (float): 估计的每分钟节拍数
    """
    from src.audio_processing import extract_pitch as _impl
    return _impl(audio_path)


def quantize_humming(pitch_data: dict) -> pretty_midi.PrettyMIDI:
    """
    将连续音高数据量化为离散MIDI音符序列。

    Args:
        pitch_data: extract_pitch 的输出字典。

    Returns:
        pretty_midi.PrettyMIDI: 量化后的MIDI对象。
    """
    from src.quantizer import quantize_humming as _impl
    return _impl(pitch_data)


def transfer_style(
    midi: pretty_midi.PrettyMIDI, style: Style
) -> pretty_midi.PrettyMIDI:
    """
    将输入MIDI的风格迁移到目标风格。

    Args:
        midi: 输入 PrettyMIDI 对象。
        style: 目标风格 ('pop'|'jazz'|'classical'|'folk')。

    Returns:
        pretty_midi.PrettyMIDI: 风格迁移后的MIDI对象。
    """
    from src.style_transfer import transfer_style as _impl
    return _impl(midi, style)


def render_audio(midi: pretty_midi.PrettyMIDI) -> str:
    """
    将MIDI渲染为WAV音频。

    Args:
        midi: 待渲染的 PrettyMIDI 对象。

    Returns:
        str: 生成的WAV文件路径。
    """
    from src.renderer import render_audio as _impl
    return _impl(midi)
