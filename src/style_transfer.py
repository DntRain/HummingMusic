"""
style_transfer.py - 风格迁移模块

将量化后的MIDI旋律迁移到目标风格，包括：
- MIDI → Piano Roll 编码（16分音符分辨率）
- VQ-VAE 风格迁移模型
- 风格参考向量加载
- music21 和弦推断与伴奏生成
- Fallback：模型未加载时直接返回原始MIDI
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)

# 加载配置
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _config = yaml.safe_load(_f)

Style = Literal["pop", "jazz", "classical", "folk"]
VALID_STYLES = ["pop", "jazz", "classical", "folk"]

# 各风格的默认音色映射 (General MIDI program number)
STYLE_PROGRAMS = {
    "pop": {"melody": 0, "chords": 4, "bass": 33},      # Piano, EP, Bass
    "jazz": {"melody": 66, "chords": 2, "bass": 32},     # Sax, EP, Acoustic Bass
    "classical": {"melody": 40, "chords": 48, "bass": 42},  # Violin, Strings, Cello
    "folk": {"melody": 25, "chords": 24, "bass": 21},    # Guitar, Nylon Guitar, Accordion
}


# ──────────────────────────────────────────────
# MIDI ↔ Piano Roll 编码
# ──────────────────────────────────────────────

def midi_to_piano_roll(
    midi: pretty_midi.PrettyMIDI, fs: int | None = None
) -> np.ndarray:
    """
    将 MIDI 转换为 piano roll 矩阵。

    Args:
        midi: PrettyMIDI 对象。
        fs: 每秒帧数。默认使用配置中的16分音符分辨率。

    Returns:
        np.ndarray: Piano roll 矩阵，shape=(128, T)，值为力度 [0, 127]。
    """
    if fs is None:
        # 根据 tempo 和分辨率计算帧率
        tempo = midi.estimate_tempo()
        resolution = _config["style_transfer"]["resolution"]
        fs = int(tempo * resolution / 60.0)
        fs = max(fs, 1)

    roll = midi.get_piano_roll(fs=fs)
    return roll


def piano_roll_to_midi(
    roll: np.ndarray,
    bpm: float,
    program: int = 0,
    instrument_name: str = "melody",
) -> pretty_midi.PrettyMIDI:
    """
    将 piano roll 矩阵转换回 MIDI。

    Args:
        roll: Piano roll 矩阵，shape=(128, T)。
        bpm: 每分钟节拍数。
        program: General MIDI 程序号。
        instrument_name: 乐器名称。

    Returns:
        pretty_midi.PrettyMIDI: MIDI 对象。
    """
    resolution = _config["style_transfer"]["resolution"]
    fs = max(int(bpm * resolution / 60.0), 1)
    frame_duration = 1.0 / fs

    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(
        program=program, is_drum=False, name=instrument_name
    )

    for pitch in range(128):
        active = roll[pitch] > 0
        if not np.any(active):
            continue

        changes = np.diff(active.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if active[0]:
            starts = np.concatenate([[0], starts])
        if active[-1]:
            ends = np.concatenate([ends, [len(active)]])

        for s, e in zip(starts, ends):
            velocity = int(np.mean(roll[pitch, s:e]))
            velocity = max(1, min(127, velocity))
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=s * frame_duration,
                end=e * frame_duration,
            )
            instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi


# ──────────────────────────────────────────────
# VQ-VAE 模型定义
# ──────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """向量量化层，用于 VQ-VAE 的离散瓶颈。"""

    def __init__(self, codebook_size: int = 512, embedding_dim: int = 64):
        """
        Args:
            codebook_size: 码本大小。
            embedding_dim: 嵌入维度。
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.data.uniform_(
            -1.0 / codebook_size, 1.0 / codebook_size
        )

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        向量量化前向传播。

        Args:
            z: 编码器输出，shape=(B, D, T)。

        Returns:
            tuple:
                - z_q: 量化后的向量，shape=(B, D, T)
                - indices: 码本索引，shape=(B, T)
                - vq_loss: VQ损失标量
        """
        # (B, D, T) -> (B, T, D)
        z_perm = z.permute(0, 2, 1).contiguous()
        flat = z_perm.view(-1, self.embedding_dim)

        # 计算距离
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1, keepdim=True).t()
        )

        indices = distances.argmin(dim=1)
        z_q = self.codebook(indices).view(z_perm.shape)

        # 损失
        commitment_loss = F.mse_loss(z_perm, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_perm.detach())
        vq_loss = codebook_loss + 0.25 * commitment_loss

        # 直通估计
        z_q = z_perm + (z_q - z_perm).detach()
        z_q = z_q.permute(0, 2, 1).contiguous()

        indices = indices.view(z.shape[0], -1)
        return z_q, indices, vq_loss


class Encoder(nn.Module):
    """三层一维卷积编码器。"""

    def __init__(
        self,
        in_channels: int = 128,
        channels: list[int] | None = None,
        embedding_dim: int = 64,
    ):
        """
        Args:
            in_channels: 输入通道数（钢琴roll的128个音高）。
            channels: 各卷积层输出通道数列表。
            embedding_dim: 最终嵌入维度。
        """
        super().__init__()
        if channels is None:
            channels = _config["style_transfer"]["encoder_channels"]

        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch

        layers.append(nn.Conv1d(prev_ch, embedding_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码 piano roll → 潜在表示。"""
        return self.net(x)


class Decoder(nn.Module):
    """三层一维转置卷积解码器。"""

    def __init__(
        self,
        out_channels: int = 128,
        channels: list[int] | None = None,
        embedding_dim: int = 64,
    ):
        """
        Args:
            out_channels: 输出通道数。
            channels: 各反卷积层通道数（逆序）。
            embedding_dim: 输入嵌入维度。
        """
        super().__init__()
        if channels is None:
            channels = list(reversed(_config["style_transfer"]["encoder_channels"]))

        layers = [nn.Conv1d(embedding_dim, channels[0], kernel_size=1)]
        prev_ch = channels[0]
        for ch in channels[1:]:
            layers.extend([
                nn.ConvTranspose1d(
                    prev_ch, ch, kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_ch = ch

        layers.extend([
            nn.ConvTranspose1d(
                prev_ch, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """解码潜在表示 → piano roll。"""
        return self.net(z_q)


class StyleVQVAE(nn.Module):
    """
    VQ-VAE 风格迁移模型。

    架构:
        - 编码器：3层一维卷积，将piano roll编码为潜在空间
        - 向量量化：离散码本瓶颈（大小512，维度64）
        - 解码器：3层一维转置卷积，重建piano roll
        - 风格条件：通过码本替换实现风格迁移
    """

    def __init__(
        self,
        in_channels: int = 128,
        codebook_size: int = 512,
        embedding_dim: int = 64,
    ):
        """
        Args:
            in_channels: piano roll通道数。
            codebook_size: VQ码本大小。
            embedding_dim: 潜在空间维度。
        """
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, embedding_dim=embedding_dim)
        self.vq = VectorQuantizer(codebook_size=codebook_size, embedding_dim=embedding_dim)
        self.decoder = Decoder(out_channels=in_channels, embedding_dim=embedding_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Args:
            x: piano roll 输入，shape=(B, 128, T)。

        Returns:
            tuple: (重建, 码本索引, VQ损失)
        """
        z = self.encoder(x)
        z_q, indices, vq_loss = self.vq(z)
        recon = self.decoder(z_q)
        return recon, indices, vq_loss

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """编码并量化。"""
        z = self.encoder(x)
        z_q, indices, _ = self.vq(z)
        return z_q, indices

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """从量化向量解码。"""
        return self.decoder(z_q)


# ──────────────────────────────────────────────
# 风格参考向量加载
# ──────────────────────────────────────────────

_style_vectors: dict[str, np.ndarray] = {}


def _load_style_vectors() -> dict[str, np.ndarray]:
    """
    从预计算的 .npy 文件加载各风格的参考向量。

    文件命名约定: {style}_vector.npy

    Returns:
        dict[str, np.ndarray]: 风格名 → 参考向量映射。
    """
    global _style_vectors  # noqa: F824
    if _style_vectors:
        return _style_vectors

    style_dir = Path(_config["style_transfer"]["style_vectors_dir"])
    for style in VALID_STYLES:
        vec_path = style_dir / f"{style}_vector.npy"
        if vec_path.exists():
            _style_vectors[style] = np.load(str(vec_path))
            logger.info("已加载风格向量: %s", vec_path)
        else:
            logger.warning("风格向量不存在: %s", vec_path)

    return _style_vectors


# ──────────────────────────────────────────────
# music21 和弦推断与伴奏生成
# ──────────────────────────────────────────────

def _infer_chords_and_add_accompaniment(
    midi: pretty_midi.PrettyMIDI, style: str
) -> pretty_midi.PrettyMIDI:
    """
    根据旋律调性推断和弦，添加和声伴奏音轨。

    使用 music21 分析旋律的调性和和声进行，生成风格化伴奏。

    Args:
        midi: 输入MIDI（包含旋律轨道）。
        style: 目标风格。

    Returns:
        pretty_midi.PrettyMIDI: 添加了伴奏轨道的MIDI。
    """
    try:
        import music21
    except ImportError:
        logger.warning("music21 不可用，跳过和弦推断")
        return midi

    if not midi.instruments or not midi.instruments[0].notes:
        return midi

    # 将旋律转为 music21 stream 进行分析
    melody_stream = music21.stream.Stream()
    for note in midi.instruments[0].notes:
        m21_note = music21.note.Note(note.pitch)
        m21_note.quarterLength = (note.end - note.start) * midi.estimate_tempo() / 60.0
        melody_stream.append(m21_note)

    # 调性分析
    key = melody_stream.analyze("key")
    logger.info("检测到调性: %s", key)

    # 根据调性生成简单和弦进行
    chord_progression = _generate_chord_progression(key, style)

    # 获取风格音色
    programs = STYLE_PROGRAMS.get(style, STYLE_PROGRAMS["pop"])

    # 添加和弦轨道
    chord_track = pretty_midi.Instrument(
        program=programs["chords"], is_drum=False, name="chords"
    )

    # 添加低音轨道
    bass_track = pretty_midi.Instrument(
        program=programs["bass"], is_drum=False, name="bass"
    )

    total_duration = midi.get_end_time()
    beat_duration = 60.0 / max(midi.estimate_tempo(), 1.0)
    beats_per_chord = 4  # 每个和弦持续4拍

    t = 0.0
    chord_idx = 0
    while t < total_duration:
        chord = chord_progression[chord_idx % len(chord_progression)]
        chord_duration = min(beat_duration * beats_per_chord, total_duration - t)

        # 和弦音
        for pitch in chord:
            chord_note = pretty_midi.Note(
                velocity=70, pitch=pitch, start=t, end=t + chord_duration
            )
            chord_track.notes.append(chord_note)

        # 低音（和弦根音低一个八度）
        bass_note = pretty_midi.Note(
            velocity=80, pitch=chord[0] - 12, start=t, end=t + chord_duration
        )
        bass_track.notes.append(bass_note)

        t += chord_duration
        chord_idx += 1

    # 修改旋律音色
    if midi.instruments:
        midi.instruments[0].program = programs["melody"]

    midi.instruments.append(chord_track)
    midi.instruments.append(bass_track)

    return midi


def _generate_chord_progression(
    key, style: str
) -> list[list[int]]:
    """
    根据调性和风格生成和弦进行。

    Args:
        key: music21 Key 对象。
        style: 目标风格。

    Returns:
        list[list[int]]: 和弦列表，每个和弦为MIDI音高列表。
    """
    tonic_pitch = key.tonic.midi

    if key.mode == "minor":
        # 小调：i - iv - v - i
        base_chords = [
            [0, 3, 7],     # i (minor)
            [5, 8, 12],    # iv (minor)
            [7, 11, 14],   # v (minor) 或 V (major)
            [0, 3, 7],     # i
        ]
    else:
        # 大调：I - IV - V - I
        base_chords = [
            [0, 4, 7],     # I (major)
            [5, 9, 12],    # IV (major)
            [7, 11, 14],   # V (major)
            [0, 4, 7],     # I
        ]

    # 风格变化
    if style == "jazz":
        # 爵士：加七音
        for chord in base_chords:
            chord.append(chord[0] + 10)  # 添加小七度
    elif style == "classical":
        # 古典：使用更传统的进行 I-V-vi-IV
        if key.mode != "minor":
            base_chords = [
                [0, 4, 7],     # I
                [7, 11, 14],   # V
                [9, 12, 16],   # vi
                [5, 9, 12],    # IV
            ]

    # 转换为绝对音高 (基于中央C附近)
    base_octave = 60  # C4
    root = tonic_pitch % 12 + base_octave - (base_octave % 12)

    chords = []
    for chord in base_chords:
        absolute_chord = [root + p for p in chord]
        # 确保在有效范围内
        absolute_chord = [max(0, min(127, p)) for p in absolute_chord]
        chords.append(absolute_chord)

    return chords


# ──────────────────────────────────────────────
# VQ-VAE 模型加载
# ──────────────────────────────────────────────

_vqvae_model: StyleVQVAE | None = None


def _load_vqvae_model() -> StyleVQVAE | None:
    """
    加载 VQ-VAE 模型权重。

    Returns:
        StyleVQVAE | None: 模型实例或 None。
    """
    global _vqvae_model
    if _vqvae_model is not None:
        return _vqvae_model

    model_path = Path(_config["style_transfer"]["model_path"])
    if not model_path.exists():
        logger.warning("VQ-VAE 模型权重不存在: %s，将使用 fallback", model_path)
        return None

    try:
        model = StyleVQVAE(
            codebook_size=_config["style_transfer"]["codebook_size"],
            embedding_dim=_config["style_transfer"]["embedding_dim"],
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        _vqvae_model = model
        logger.info("VQ-VAE 模型加载成功: %s", model_path)
        return model
    except Exception as e:
        logger.error("VQ-VAE 模型加载失败: %s", e)
        return None


# ──────────────────────────────────────────────
# Fallback: 直接返回带伴奏的原始MIDI
# ──────────────────────────────────────────────

def _fallback_transfer(
    midi: pretty_midi.PrettyMIDI, style: str
) -> pretty_midi.PrettyMIDI:
    """
    Fallback 风格迁移：不改变旋律，仅添加风格化伴奏和音色。

    Args:
        midi: 输入MIDI。
        style: 目标风格。

    Returns:
        pretty_midi.PrettyMIDI: 添加了伴奏的MIDI。
    """
    logger.warning("使用 fallback 风格迁移（模型未加载），仅添加伴奏")
    return _infer_chords_and_add_accompaniment(midi, style)


# ──────────────────────────────────────────────
# 公开接口
# ──────────────────────────────────────────────

def transfer_style(
    midi: pretty_midi.PrettyMIDI, style: Style
) -> pretty_midi.PrettyMIDI:
    """
    将输入MIDI的风格迁移到目标风格。

    处理流程:
    1. 验证风格参数
    2. 尝试加载 VQ-VAE 模型和风格向量
    3. 若模型可用：编码旋律 → 风格条件替换 → 解码 → 添加伴奏
    4. 若模型不可用：fallback 到仅添加伴奏

    Args:
        midi: 输入 PrettyMIDI 对象。
        style: 目标风格 ('pop'|'jazz'|'classical'|'folk')。

    Returns:
        pretty_midi.PrettyMIDI: 风格迁移后的MIDI对象。

    Raises:
        ValueError: style 不在有效范围内。
    """
    if style not in VALID_STYLES:
        raise ValueError(
            f"无效的风格 '{style}'，有效选项: {VALID_STYLES}"
        )

    logger.info("开始风格迁移: 目标风格=%s", style)

    model = _load_vqvae_model()
    style_vectors = _load_style_vectors()

    if model is None or style not in style_vectors:
        return _fallback_transfer(midi, style)

    try:
        # 编码
        roll = midi_to_piano_roll(midi)
        x = torch.from_numpy(roll).float().unsqueeze(0)  # (1, 128, T)

        with torch.no_grad():
            z_q, indices = model.encode(x)

            # 风格条件注入：将风格向量加到潜在表示上
            style_vec = torch.from_numpy(style_vectors[style]).float()
            style_vec = style_vec.unsqueeze(0).unsqueeze(-1)  # (1, D, 1)
            z_styled = z_q + style_vec.expand_as(z_q)

            # 解码
            recon = model.decode(z_styled)

        # 转回MIDI
        recon_roll = (recon.squeeze(0).numpy() * 127).clip(0, 127)
        bpm = midi.estimate_tempo()
        result = piano_roll_to_midi(recon_roll, bpm)

        # 添加伴奏
        result = _infer_chords_and_add_accompaniment(result, style)

        logger.info("风格迁移完成")
        return result

    except Exception as e:
        logger.error("VQ-VAE 推理失败: %s，回退到 fallback", e)
        return _fallback_transfer(midi, style)
