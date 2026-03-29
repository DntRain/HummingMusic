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
        return _baseline_quantize(pitch_data)

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
