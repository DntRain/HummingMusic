"""
renderer.py - 渲染模块

将MIDI对象渲染为可播放的WAV音频文件，包括：
- FluidSynth 合成（主方案）
- pretty_midi 内置合成（fallback）
- 临时文件管理（UUID子目录）
"""

import logging
import uuid
from pathlib import Path

import numpy as np
import pretty_midi
import yaml

logger = logging.getLogger(__name__)

# 加载配置
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _config = yaml.safe_load(_f)


def _ensure_tmp_dir() -> Path:
    """
    创建并返回一个唯一的临时输出目录。

    Returns:
        Path: 临时目录路径 (tmp/{uuid}/)。
    """
    base_tmp = Path(_config["tmp_dir"])
    request_dir = base_tmp / str(uuid.uuid4())
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def _render_with_fluidsynth(
    midi: pretty_midi.PrettyMIDI, output_path: str
) -> bool:
    """
    使用 FluidSynth 将 MIDI 渲染为 WAV。

    Args:
        midi: PrettyMIDI 对象。
        output_path: 输出 WAV 文件路径。

    Returns:
        bool: 渲染成功返回 True，失败返回 False。
    """
    soundfont_path = _config["renderer"]["soundfont_path"]
    sample_rate = _config["renderer"]["sample_rate"]

    if not Path(soundfont_path).exists():
        logger.warning("音色库文件不存在: %s", soundfont_path)
        return False

    try:
        import fluidsynth

        # 先保存为临时MIDI文件
        tmp_midi_path = str(Path(output_path).with_suffix(".mid"))
        midi.write(tmp_midi_path)

        # 使用 FluidSynth 合成
        fs = fluidsynth.Synth(samplerate=float(sample_rate))
        sfid = fs.sfload(soundfont_path)
        fs.program_select(0, sfid, 0, 0)

        # 通过 MIDI 文件合成
        # 使用 fluidsynth 的命令行方式渲染
        import subprocess

        result = subprocess.run(
            [
                "fluidsynth",
                "-ni",
                soundfont_path,
                tmp_midi_path,
                "-F", output_path,
                "-r", str(sample_rate),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # 清理临时MIDI文件
        Path(tmp_midi_path).unlink(missing_ok=True)

        if result.returncode == 0 and Path(output_path).exists():
            logger.info("FluidSynth 渲染成功: %s", output_path)
            return True
        else:
            logger.warning("FluidSynth 渲染失败: %s", result.stderr)
            return False

    except (ImportError, FileNotFoundError):
        logger.warning("FluidSynth 不可用")
        return False
    except Exception as e:
        logger.warning("FluidSynth 渲染异常: %s", e)
        return False


def _render_with_pretty_midi(
    midi: pretty_midi.PrettyMIDI, output_path: str
) -> bool:
    """
    使用 pretty_midi 内置方法将 MIDI 渲染为 WAV（fallback）。

    使用正弦波合成，音质较低但无需外部依赖。

    Args:
        midi: PrettyMIDI 对象。
        output_path: 输出 WAV 文件路径。

    Returns:
        bool: 渲染成功返回 True，失败返回 False。
    """
    try:
        import soundfile as sf

        sample_rate = _config["renderer"]["sample_rate"]
        audio = midi.synthesize(fs=sample_rate)

        # 归一化到 [-1, 1]
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        sf.write(output_path, audio, sample_rate, subtype="PCM_16")
        logger.info("pretty_midi 合成渲染成功 (fallback): %s", output_path)
        return True

    except Exception as e:
        logger.error("pretty_midi 渲染也失败了: %s", e)
        return False


def render_audio(midi: pretty_midi.PrettyMIDI) -> str:
    """
    将MIDI对象渲染为可播放的WAV音频文件。

    渲染策略:
    1. 优先使用 FluidSynth（高质量 SoundFont 合成）
    2. FluidSynth 不可用时 fallback 到 pretty_midi 内置合成

    输出文件写入 tmp/{uuid}/ 目录，避免并发冲突。

    Args:
        midi: 待渲染的 PrettyMIDI 对象。

    Returns:
        str: 生成的 WAV 文件绝对路径。

    Raises:
        RuntimeError: 所有渲染方法均失败。
    """
    logger.info("开始音频渲染")

    # 创建输出目录
    tmp_dir = _ensure_tmp_dir()
    output_path = str(tmp_dir / "output.wav")

    # 同时保存MIDI文件供下载
    midi_path = str(tmp_dir / "output.mid")
    midi.write(midi_path)
    logger.info("MIDI 文件已保存: %s", midi_path)

    # 尝试 FluidSynth
    if _render_with_fluidsynth(midi, output_path):
        return output_path

    # Fallback 到 pretty_midi
    if _render_with_pretty_midi(midi, output_path):
        return output_path

    raise RuntimeError("所有渲染方法均失败，无法生成音频")
