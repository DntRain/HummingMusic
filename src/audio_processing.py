"""
audio_processing.py - 音频处理模块

负责从输入音频中提取基频(F0)信息，包括：
- 音频重采样与格式标准化
- 使用 CREPE 进行音高检测（Viterbi 平滑）
- 低置信度帧过滤
- 短静音段线性插值修复
- BPM 估计
"""

import logging
from pathlib import Path

import librosa
import numpy as np
import yaml

try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    crepe = None
    CREPE_AVAILABLE = False

logger = logging.getLogger(__name__)

# 加载配置
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _config = yaml.safe_load(_f)


def _load_and_resample(audio_path: str) -> tuple[np.ndarray, int]:
    """
    加载音频文件并重采样为标准格式。

    Args:
        audio_path: 音频文件路径。

    Returns:
        tuple: (audio_signal, sample_rate)
            - audio_signal (np.ndarray): 单声道音频信号，shape=(N,)
            - sample_rate (int): 采样率（固定为配置值）

    Raises:
        FileNotFoundError: 音频文件不存在。
        RuntimeError: 音频文件无法读取。
    """
    audio_path = str(audio_path)
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    target_sr = _config["audio"]["sample_rate"]

    try:
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"无法读取音频文件 {audio_path}: {e}") from e

    logger.info(
        "已加载音频: %s, 时长=%.2fs, 采样率=%dHz",
        audio_path, len(y) / sr, sr,
    )
    return y, sr


def _extract_f0(
    audio: np.ndarray, sr: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 CREPE 提取基频。

    Args:
        audio: 单声道音频信号。
        sr: 采样率。

    Returns:
        tuple: (time, frequency, confidence)
            - time (np.ndarray): 时间戳，shape=(N,)
            - frequency (np.ndarray): F0频率(Hz)，shape=(N,)
            - confidence (np.ndarray): 置信度[0,1]，shape=(N,)
    """
    if not CREPE_AVAILABLE:
        raise RuntimeError("crepe 未安装，无法提取音高")

    step_size = _config["crepe"]["step_size"]
    model_capacity = _config["crepe"]["model_capacity"]
    viterbi = _config["crepe"]["viterbi"]

    time, frequency, confidence, _ = crepe.predict(
        audio,
        sr,
        model_capacity=model_capacity,
        viterbi=viterbi,
        step_size=step_size,
    )

    logger.info("CREPE 提取完成: %d 帧", len(time))
    return time, frequency, confidence


def _filter_low_confidence(
    frequency: np.ndarray, confidence: np.ndarray
) -> np.ndarray:
    """
    将低置信度帧的频率置为 NaN。

    Args:
        frequency: F0频率数组。
        confidence: 置信度数组。

    Returns:
        np.ndarray: 过滤后的频率数组，低置信度帧为 NaN。
    """
    threshold = _config["crepe"]["confidence_threshold"]
    filtered = frequency.copy()
    low_conf_mask = confidence < threshold
    filtered[low_conf_mask] = np.nan

    n_filtered = np.sum(low_conf_mask)
    logger.info(
        "置信度过滤: 阈值=%.2f, 过滤 %d/%d 帧 (%.1f%%)",
        threshold, n_filtered, len(frequency),
        100.0 * n_filtered / max(len(frequency), 1),
    )
    return filtered


def _interpolate_short_gaps(
    time: np.ndarray, frequency: np.ndarray
) -> np.ndarray:
    """
    对短暂 NaN 段做线性插值，长静音段保留 NaN。

    短暂段定义：连续 NaN 帧的时间跨度 < max_nan_duration。

    Args:
        time: 时间戳数组。
        frequency: 频率数组（含NaN）。

    Returns:
        np.ndarray: 插值修复后的频率数组。
    """
    max_nan_dur = _config["interpolation"]["max_nan_duration"]
    result = frequency.copy()
    nan_mask = np.isnan(result)

    if not np.any(nan_mask):
        return result

    # 找到连续NaN段的起止索引
    changes = np.diff(nan_mask.astype(int))
    starts = np.where(changes == 1)[0] + 1   # NaN段起始
    ends = np.where(changes == -1)[0] + 1     # NaN段结束

    # 处理边界情况
    if nan_mask[0]:
        starts = np.concatenate([[0], starts])
    if nan_mask[-1]:
        ends = np.concatenate([ends, [len(frequency)]])

    n_interpolated = 0
    for s, e in zip(starts, ends):
        gap_duration = time[min(e, len(time) - 1)] - time[s]
        if gap_duration < max_nan_dur and s > 0 and e < len(frequency):
            # 线性插值
            result[s:e] = np.interp(
                time[s:e],
                [time[s - 1], time[e]],
                [result[s - 1], result[e]],
            )
            n_interpolated += 1

    logger.info(
        "插值修复: %d 个短NaN段 (阈值=%.2fs)", n_interpolated, max_nan_dur
    )
    return result


def _estimate_bpm(audio: np.ndarray, sr: int) -> float:
    """
    使用 librosa 估计 BPM。

    Args:
        audio: 单声道音频信号。
        sr: 采样率。

    Returns:
        float: 估计的BPM值。
    """
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    # librosa >= 0.10 返回数组
    bpm = float(np.atleast_1d(tempo)[0])
    logger.info("BPM 估计: %.1f", bpm)
    return bpm


def extract_pitch(audio_path: str) -> dict:
    """
    从音频文件中提取基频(F0)信息。

    完整处理流程:
    1. 加载并重采样音频至 16kHz 单声道
    2. 使用 CREPE (Viterbi平滑) 提取F0
    3. 过滤低置信度帧（阈值由 config.yaml 控制）
    4. 对短暂 NaN 段做线性插值
    5. 估计 BPM

    Args:
        audio_path: 输入音频文件路径，支持 WAV/MP3 格式。

    Returns:
        dict: 包含以下字段:
            - time (np.ndarray): 时间戳数组，单位秒，shape=(N,)
            - frequency (np.ndarray): 基频数组，单位Hz，shape=(N,)。
              低置信度帧值为 NaN，短NaN段已插值。
            - confidence (np.ndarray): 置信度数组，范围[0,1]，shape=(N,)
            - bpm (float): 估计的每分钟节拍数

    Raises:
        FileNotFoundError: 音频文件不存在。
        RuntimeError: 音频处理失败。
    """
    logger.info("开始音高提取: %s", audio_path)

    # 1. 加载并重采样
    audio, sr = _load_and_resample(audio_path)

    # 2. 提取F0
    time, frequency, confidence = _extract_f0(audio, sr)

    # 3. 过滤低置信度
    frequency = _filter_low_confidence(frequency, confidence)

    # 4. 短NaN段插值
    frequency = _interpolate_short_gaps(time, frequency)

    # 5. 估计BPM
    bpm = _estimate_bpm(audio, sr)

    logger.info("音高提取完成: %d 帧, BPM=%.1f", len(time), bpm)

    return {
        "time": time,
        "frequency": frequency,
        "confidence": confidence,
        "bpm": bpm,
    }
