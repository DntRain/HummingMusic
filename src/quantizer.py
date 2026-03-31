"""
quantizer.py - 容错量化模块

将连续音高(F0)数据量化为离散MIDI音符序列，包括：
- F0 → MIDI 音符编号转换
- BiLSTM-CRF 序列标注模型（BIO标注方案）
- BIO标注后处理：合并连续BI段为音符事件
- Fallback：模型未加载时使用直接四舍五入方法
"""

import logging
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

# 加载配置
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _config = yaml.safe_load(_f)

# BIO 标签映射
BIO_TAGS = {"O": 0, "B": 1, "I": 2}
IDX_TO_TAG = {v: k for k, v in BIO_TAGS.items()}
NUM_TAGS = len(BIO_TAGS)


# ──────────────────────────────────────────────
# 预处理函数
# ──────────────────────────────────────────────

def _freq_to_midi(frequency: np.ndarray) -> np.ndarray:
    """
    将频率(Hz)转换为连续MIDI音符编号。

    Args:
        frequency: 频率数组，NaN 表示静音。

    Returns:
        np.ndarray: 连续MIDI编号，NaN保持不变。
            MIDI编号 = 69 + 12 * log2(freq / 440)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        midi_notes = 69.0 + 12.0 * np.log2(frequency / 440.0)
    return midi_notes


def _compute_beat_positions(
    time: np.ndarray, bpm: float
) -> np.ndarray:
    """
    计算每帧相对于节拍的位置（0~1之间的小数）。

    Args:
        time: 时间戳数组（秒）。
        bpm: 每分钟节拍数。

    Returns:
        np.ndarray: 相对节拍位置，shape=(N,)，范围 [0, 1)。
    """
    beat_duration = 60.0 / max(bpm, 1.0)
    beat_positions = (time % beat_duration) / beat_duration
    return beat_positions


def _prepare_features(pitch_data: dict) -> np.ndarray:
    """
    将音高数据转换为模型输入特征。

    特征维度:
        - MIDI音符编号（连续值，NaN替换为0）
        - 是否有效音高（0或1）
        - 置信度
        - 相对节拍位置

    Args:
        pitch_data: extract_pitch 输出字典。

    Returns:
        np.ndarray: 特征矩阵，shape=(N, 4)。
    """
    freq = pitch_data["frequency"]
    midi_notes = _freq_to_midi(freq)
    beat_pos = _compute_beat_positions(pitch_data["time"], pitch_data["bpm"])

    valid_mask = ~np.isnan(midi_notes)
    midi_filled = np.where(valid_mask, midi_notes, 0.0)

    features = np.stack(
        [
            midi_filled,
            valid_mask.astype(np.float32),
            pitch_data["confidence"],
            beat_pos,
        ],
        axis=-1,
    )
    return features.astype(np.float32)


# ──────────────────────────────────────────────
# CRF 层
# ──────────────────────────────────────────────

class CRFLayer(nn.Module):
    """
    条件随机场(CRF)解码层。

    用于序列标注任务，对标签转移概率进行建模。
    """

    def __init__(self, num_tags: int):
        """
        Args:
            num_tags: 标签数量。
        """
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(
        self, emissions: torch.Tensor, mask: torch.Tensor | None = None
    ) -> list[list[int]]:
        """
        Viterbi 解码，返回最优标签序列。

        Args:
            emissions: 发射分数，shape=(batch, seq_len, num_tags)。
            mask: 有效位掩码，shape=(batch, seq_len)。

        Returns:
            list[list[int]]: 每个样本的最优标签序列。
        """
        batch_size, seq_len, _ = emissions.shape
        if mask is None:
            mask = emissions.new_ones((batch_size, seq_len), dtype=torch.bool)

        best_paths = []
        for i in range(batch_size):
            path = self._viterbi_decode(emissions[i], mask[i])
            best_paths.append(path)
        return best_paths

    def _viterbi_decode(
        self, emission: torch.Tensor, mask: torch.Tensor
    ) -> list[int]:
        """单样本 Viterbi 解码。"""
        seq_len = int(mask.sum().item())
        emission = emission[:seq_len]

        score = self.start_transitions + emission[0]
        history = []

        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(1)
            broadcast_emission = emission[t].unsqueeze(0)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=0)
            history.append(indices)
            score = next_score

        score += self.end_transitions
        _, best_last_tag = score.max(dim=0)
        best_last_tag = best_last_tag.item()

        best_path = [best_last_tag]
        for indices in reversed(history):
            best_path.append(indices[best_path[-1]].item())
        best_path.reverse()

        return best_path


# ──────────────────────────────────────────────
# BiLSTM-CRF 模型
# ──────────────────────────────────────────────

class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF 序列标注模型。

    用于将连续音高帧序列标注为 BIO 标签，
    识别音符的起始(B)、持续(I)和静音(O)。

    架构:
        - 两层双向 LSTM (hidden_size=128)
        - Dropout=0.3
        - 全连接层映射到标签空间
        - CRF 解码层
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_tags: int = NUM_TAGS,
    ):
        """
        Args:
            input_dim: 输入特征维度。
            hidden_size: LSTM 隐藏层大小。
            num_layers: LSTM 层数。
            dropout: Dropout 概率。
            num_tags: BIO标签数量。
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_tags)
        self.crf = CRFLayer(num_tags)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> list[list[int]]:
        """
        前向推理，返回 BIO 标签序列。

        Args:
            x: 输入特征，shape=(batch, seq_len, input_dim)。
            mask: 有效位掩码，shape=(batch, seq_len)。

        Returns:
            list[list[int]]: 每个样本的 BIO 标签序列。
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        emissions = self.fc(lstm_out)
        return self.crf(emissions, mask)


# ──────────────────────────────────────────────
# BIO 后处理
# ──────────────────────────────────────────────

def _bio_to_notes(
    tags: list[int],
    time: np.ndarray,
    midi_notes: np.ndarray,
) -> list[dict]:
    """
    将 BIO 标签序列转换为音符事件列表。

    Args:
        tags: BIO标签索引列表，长度=N。
        time: 时间戳数组，shape=(N,)。
        midi_notes: 连续MIDI编号数组，shape=(N,)。

    Returns:
        list[dict]: 音符事件列表，每个事件包含:
            - pitch (int): MIDI音高（四舍五入到整数）
            - start (float): 起始时间（秒）
            - end (float): 结束时间（秒）
            - velocity (int): 力度（固定100）
    """
    notes = []
    current_note = None

    for i, tag_idx in enumerate(tags):
        tag = IDX_TO_TAG[tag_idx]

        if tag == "B":
            # 保存前一个音符
            if current_note is not None:
                current_note["end"] = time[i]
                notes.append(current_note)
            # 开始新音符
            if not np.isnan(midi_notes[i]):
                current_note = {
                    "pitch": int(round(midi_notes[i])),
                    "start": time[i],
                    "end": None,
                    "velocity": 100,
                    "_pitches": [midi_notes[i]],
                }
            else:
                current_note = None

        elif tag == "I" and current_note is not None:
            # 延续当前音符
            if not np.isnan(midi_notes[i]):
                current_note["_pitches"].append(midi_notes[i])

        else:  # O 或无效的 I（没有对应的 B）
            if current_note is not None:
                current_note["end"] = time[i]
                notes.append(current_note)
                current_note = None

    # 处理最后一个音符
    if current_note is not None:
        current_note["end"] = time[-1] if len(time) > 0 else 0.0
        notes.append(current_note)

    # 用音符段内的中位音高作为最终音高
    for note in notes:
        if "_pitches" in note and note["_pitches"]:
            note["pitch"] = int(round(np.median(note["_pitches"])))
        note.pop("_pitches", None)
        # 确保音高在有效范围内
        note["pitch"] = max(0, min(127, note["pitch"]))

    return notes


def _notes_to_midi(notes: list[dict], bpm: float) -> pretty_midi.PrettyMIDI:
    """
    将音符事件列表转换为 PrettyMIDI 对象。

    Args:
        notes: 音符事件列表。
        bpm: 每分钟节拍数。

    Returns:
        pretty_midi.PrettyMIDI: MIDI对象。
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(
        program=0, is_drum=False, name="melody"
    )

    for note_info in notes:
        if note_info["end"] is None or note_info["end"] <= note_info["start"]:
            continue
        note = pretty_midi.Note(
            velocity=note_info["velocity"],
            pitch=note_info["pitch"],
            start=note_info["start"],
            end=note_info["end"],
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi


# ──────────────────────────────────────────────
# Fallback: 四舍五入基线方法
# ──────────────────────────────────────────────

def _baseline_quantize(pitch_data: dict) -> pretty_midi.PrettyMIDI:
    """
    基线量化方法：直接将F0四舍五入到最近的MIDI音符。

    当 BiLSTM-CRF 模型权重未加载时使用此方法作为 fallback。
    使用简单的帧级聚合策略：将相邻同音高帧合并为单个音符。

    Args:
        pitch_data: extract_pitch 输出字典。

    Returns:
        pretty_midi.PrettyMIDI: 量化后的MIDI对象。
    """
    logger.warning("使用 fallback 基线量化方法（模型未加载）")

    time = pitch_data["time"]
    freq = pitch_data["frequency"]
    bpm = pitch_data["bpm"]

    midi_notes = _freq_to_midi(freq)

    # 四舍五入到最近整数
    rounded = np.where(np.isnan(midi_notes), np.nan, np.round(midi_notes))

    # 相邻同音高帧合并
    notes = []
    current_pitch = None
    start_time = None

    for i in range(len(rounded)):
        pitch = rounded[i]

        if np.isnan(pitch):
            if current_pitch is not None:
                notes.append({
                    "pitch": int(current_pitch),
                    "start": start_time,
                    "end": time[i],
                    "velocity": 100,
                })
                current_pitch = None
        elif current_pitch is None or pitch != current_pitch:
            if current_pitch is not None:
                notes.append({
                    "pitch": int(current_pitch),
                    "start": start_time,
                    "end": time[i],
                    "velocity": 100,
                })
            current_pitch = pitch
            start_time = time[i]

    # 最后一个音符
    if current_pitch is not None:
        notes.append({
            "pitch": int(current_pitch),
            "start": start_time,
            "end": time[-1],
            "velocity": 100,
        })

    return _notes_to_midi(notes, bpm)


# ──────────────────────────────────────────────
# RoundingBaselineQuantizer（正式 baseline 类）
# ──────────────────────────────────────────────

class RoundingBaselineQuantizer:
    """
    四舍五入基线量化器。

    将连续 F0 频率直接四舍五入到最近半音（整数 MIDI 编号），
    作为后续模型对比实验的基准线。

    处理流程:
        1. 置信度过滤：低于阈值的帧标记为静音 (NaN)
        2. 短静音插值：短于阈值的静音段用线性插值填充
        3. F0 → MIDI 转换并四舍五入
        4. 相邻同音高帧合并为单个音符
        5. 输出标准 PrettyMIDI 对象

    所有阈值从 config.yaml 的 quantizer 节读取，不硬编码。
    无需预训练权重，可独立运行。

    实现 interfaces.HummingQuantizer 协议。
    """

    def __init__(self) -> None:
        """初始化，从配置文件读取阈值参数。"""
        self._confidence_threshold: float = _config["quantizer"].get(
            "confidence_threshold", 0.8
        )
        self._silence_threshold: float = _config["quantizer"].get(
            "silence_threshold", 0.2
        )
        self._velocity: int = _config["quantizer"].get(
            "default_velocity", 80
        )

    def __call__(self, pitch_data: dict) -> pretty_midi.PrettyMIDI:
        """
        将连续音高数据量化为离散 MIDI 音符序列。

        Args:
            pitch_data: extract_pitch 的输出字典，包含:
                - time (np.ndarray): 时间戳数组，单位秒，shape=(N,)
                - frequency (np.ndarray): 基频数组，单位Hz，shape=(N,)
                - confidence (np.ndarray): 置信度数组，范围[0,1]，shape=(N,)
                - bpm (float): 估计的每分钟节拍数

        Returns:
            pretty_midi.PrettyMIDI: 量化后的MIDI对象，包含单个旋律轨道，
                力度统一为 config 中的 default_velocity。
        """
        time = pitch_data["time"]
        frequency = pitch_data["frequency"].copy()
        confidence = pitch_data["confidence"]
        bpm = pitch_data["bpm"]

        logger.info(
            "RoundingBaselineQuantizer: %d 帧, BPM=%.1f",
            len(time), bpm,
        )

        # 1. 置信度过滤
        frequency = self._filter_by_confidence(frequency, confidence)

        # 2. 短静音段线性插值
        frequency = self._interpolate_short_silences(time, frequency)

        # 3. F0 → MIDI → 四舍五入
        midi_continuous = _freq_to_midi(frequency)
        midi_rounded = np.where(
            np.isnan(midi_continuous), np.nan, np.round(midi_continuous)
        )
        # 钳制到有效 MIDI 范围 [0, 127]
        midi_rounded = np.where(
            np.isnan(midi_rounded), np.nan,
            np.clip(midi_rounded, 0, 127),
        )

        # 4. 相邻同音高帧合并
        notes = self._merge_frames_to_notes(time, midi_rounded)

        logger.info("RoundingBaselineQuantizer: 输出 %d 个音符", len(notes))

        # 5. 转为 PrettyMIDI
        return _notes_to_midi(notes, bpm)

    def _filter_by_confidence(
        self, frequency: np.ndarray, confidence: np.ndarray
    ) -> np.ndarray:
        """
        将置信度低于阈值的帧频率置为 NaN。

        Args:
            frequency: 基频数组 (Hz)，shape=(N,)。
            confidence: 置信度数组 [0,1]，shape=(N,)。

        Returns:
            np.ndarray: 过滤后的频率数组，低置信度帧为 NaN。
        """
        result = frequency.copy()
        mask = confidence < self._confidence_threshold
        result[mask] = np.nan
        n_filtered = int(np.sum(mask))
        if n_filtered > 0:
            logger.debug(
                "置信度过滤: %d/%d 帧 (阈值=%.2f)",
                n_filtered, len(frequency), self._confidence_threshold,
            )
        return result

    def _interpolate_short_silences(
        self, time: np.ndarray, frequency: np.ndarray
    ) -> np.ndarray:
        """
        对短于阈值的静音段(NaN)做线性插值，长静音段保留 NaN。

        Args:
            time: 时间戳数组 (秒)，shape=(N,)。
            frequency: 频率数组（含NaN），shape=(N,)。

        Returns:
            np.ndarray: 插值修复后的频率数组。
        """
        result = frequency.copy()
        nan_mask = np.isnan(result)

        if not np.any(nan_mask) or np.all(nan_mask):
            return result

        # 找连续 NaN 段的起止索引
        changes = np.diff(nan_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if nan_mask[0]:
            starts = np.concatenate([[0], starts])
        if nan_mask[-1]:
            ends = np.concatenate([ends, [len(frequency)]])

        n_interpolated = 0
        for s, e in zip(starts, ends):
            gap_duration = time[min(e, len(time) - 1)] - time[s]
            # 仅对短静音段插值，且两端必须有有效值
            if gap_duration < self._silence_threshold and s > 0 and e < len(frequency):
                result[s:e] = np.interp(
                    time[s:e],
                    [time[s - 1], time[e]],
                    [result[s - 1], result[e]],
                )
                n_interpolated += 1

        if n_interpolated > 0:
            logger.debug(
                "短静音插值: %d 段 (阈值=%.2fs)",
                n_interpolated, self._silence_threshold,
            )
        return result

    def _merge_frames_to_notes(
        self, time: np.ndarray, midi_rounded: np.ndarray
    ) -> list[dict]:
        """
        将逐帧 MIDI 编号合并为音符事件列表。

        相邻帧若为同一音高则合并为单个音符；
        NaN 帧视为音符间隔。

        Args:
            time: 时间戳数组 (秒)，shape=(N,)。
            midi_rounded: 四舍五入后的 MIDI 编号数组，shape=(N,)。

        Returns:
            list[dict]: 音符事件列表，每个事件包含:
                - pitch (int): MIDI 音高 [0, 127]
                - start (float): 起始时间（秒）
                - end (float): 结束时间（秒）
                - velocity (int): 力度
        """
        notes: list[dict] = []
        current_pitch: float | None = None
        start_time: float = 0.0

        for i in range(len(midi_rounded)):
            pitch = midi_rounded[i]

            if np.isnan(pitch):
                # 静音帧：结束当前音符
                if current_pitch is not None:
                    notes.append({
                        "pitch": int(current_pitch),
                        "start": start_time,
                        "end": time[i],
                        "velocity": self._velocity,
                    })
                    current_pitch = None
            elif current_pitch is None or pitch != current_pitch:
                # 新音高：结束前一个音符，开始新音符
                if current_pitch is not None:
                    notes.append({
                        "pitch": int(current_pitch),
                        "start": start_time,
                        "end": time[i],
                        "velocity": self._velocity,
                    })
                current_pitch = pitch
                start_time = time[i]
            # else: 同一音高，继续延长

        # 处理序列末尾的音符
        if current_pitch is not None and len(time) > 0:
            # 末帧时间 + 一个帧间隔作为结束时间
            if len(time) >= 2:
                frame_dur = time[-1] - time[-2]
            else:
                frame_dur = 0.01  # 单帧时给一个最小时长
            notes.append({
                "pitch": int(current_pitch),
                "start": start_time,
                "end": time[-1] + frame_dur,
                "velocity": self._velocity,
            })

        return notes


# ──────────────────────────────────────────────
# 模型加载与推理
# ──────────────────────────────────────────────

_model: BiLSTMCRF | None = None


def _load_model() -> BiLSTMCRF | None:
    """
    从配置路径加载 BiLSTM-CRF 模型权重。

    Returns:
        BiLSTMCRF | None: 加载成功返回模型实例，失败返回 None。
    """
    global _model
    if _model is not None:
        return _model

    model_path = Path(_config["quantizer"]["model_path"])
    if not model_path.exists():
        logger.warning("量化模型权重不存在: %s，将使用 fallback", model_path)
        return None

    try:
        model = BiLSTMCRF(
            input_dim=4,
            hidden_size=_config["quantizer"]["hidden_size"],
            num_layers=_config["quantizer"]["num_layers"],
            dropout=_config["quantizer"]["dropout"],
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        _model = model
        logger.info("量化模型加载成功: %s", model_path)
        return model
    except Exception as e:
        logger.error("量化模型加载失败: %s", e)
        return None


# ──────────────────────────────────────────────
# 公开接口
# ──────────────────────────────────────────────

def quantize_humming(pitch_data: dict) -> pretty_midi.PrettyMIDI:
    """
    将连续音高数据量化为离散MIDI音符序列。

    处理流程:
    1. 尝试加载 BiLSTM-CRF 模型
    2. 若模型可用：提取特征 → 模型推理 → BIO后处理 → 生成MIDI
    3. 若模型不可用：fallback 到四舍五入基线方法

    Args:
        pitch_data: extract_pitch 的输出字典，包含:
            - time (np.ndarray): 时间戳数组
            - frequency (np.ndarray): 基频数组（含NaN）
            - confidence (np.ndarray): 置信度数组
            - bpm (float): BPM

    Returns:
        pretty_midi.PrettyMIDI: 量化后的MIDI对象，包含单个旋律轨道。
    """
    logger.info("开始哼唱量化")

    model = _load_model()
    if model is None:
        baseline = RoundingBaselineQuantizer()
        return baseline(pitch_data)

    # 准备特征
    features = _prepare_features(pitch_data)
    x = torch.from_numpy(features).unsqueeze(0)  # (1, seq_len, 4)

    # 推理
    with torch.no_grad():
        tag_sequences = model(x)

    tags = tag_sequences[0]

    # BIO → 音符事件
    midi_notes = _freq_to_midi(pitch_data["frequency"])
    notes = _bio_to_notes(tags, pitch_data["time"], midi_notes)

    logger.info("量化完成: %d 个音符", len(notes))

    return _notes_to_midi(notes, pitch_data["bpm"])
