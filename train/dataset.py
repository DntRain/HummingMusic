"""
train/dataset.py - HumTrans 数据集类

将 HumTrans 数据集封装为 PyTorch Dataset，
支持：
- 从预提取的 .npy 特征文件加载（快，训练用）
- 从原始 WAV 实时提取（慢，调试用）

标注转换：将 MIDI 音符映射为帧级 BIO 标签。
    B (1)：音符起始帧
    I (2)：音符持续帧
    O (0)：静音帧
"""

import json
import logging
from pathlib import Path

import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# BIO 标签
BIO_O = 0
BIO_B = 1
BIO_I = 2

FRAME_STEP_MS = 10          # 与 CREPE step_size 一致
FRAME_STEP_S = FRAME_STEP_MS / 1000.0


class HumTransDataset(Dataset):
    """
    HumTrans 哼唱转录数据集。

    目录结构（由 __init__ 参数指定）：
        wav_dir/    *.wav
        midi_dir/   *.mid
        feat_dir/   *.npy  （预提取特征，可选）

    Args:
        split_json: train_valid_test_keys.json 路径。
        split: 'TRAIN' | 'VALID' | 'TEST'。
        wav_dir: WAV 文件目录。
        midi_dir: MIDI 文件目录。
        feat_dir: 预提取 .npy 特征目录（None 则实时提取）。
        max_samples: 限制样本数（None 表示全量）。
    """

    def __init__(
        self,
        split_json: str,
        split: str,
        wav_dir: str,
        midi_dir: str,
        feat_dir: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        split = split.upper()
        assert split in ("TRAIN", "VALID", "TEST"), f"无效 split: {split}"

        with open(split_json, "r", encoding="utf-8") as f:
            keys = json.load(f)[split]

        if max_samples is not None:
            keys = keys[:max_samples]

        self.keys = keys
        self.wav_dir = Path(wav_dir)
        self.midi_dir = Path(midi_dir)
        self.feat_dir = Path(feat_dir) if feat_dir else None

        # 验证文件是否存在，过滤掉缺失的样本
        valid_keys = []
        for key in self.keys:
            wav = self.wav_dir / f"{key}.wav"
            mid = self.midi_dir / f"{key}.mid"
            if wav.exists() and mid.exists():
                valid_keys.append(key)
            else:
                logger.warning("跳过缺失样本: %s", key)

        self.keys = valid_keys
        logger.info("HumTransDataset [%s]: %d 条样本", split, len(self.keys))

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict:
        """
        返回单条样本。

        Returns:
            dict:
                - key (str): 样本 ID
                - features (Tensor): shape=(T, 4)，float32
                - labels (Tensor): shape=(T,)，int64，BIO 标签
                - n_frames (int): 有效帧数 T
        """
        key = self.keys[idx]
        features = self._load_features(key)       # (T, 4) numpy
        labels = self._load_labels(key, n_frames=len(features), features=features)  # (T,) numpy

        return {
            "key": key,
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "n_frames": len(features),
        }

    # ──────────────────────────────────────────────
    # 特征加载
    # ──────────────────────────────────────────────

    def _load_features(self, key: str) -> np.ndarray:
        """
        加载帧级特征，优先读取预提取 .npy，否则实时提取。
        自动检测并修正演唱者低/高八度哼唱导致的音高偏移。

        Returns:
            np.ndarray: shape=(T, 4)，float32。
                列: [midi_note, is_valid, confidence, beat_position]
        """
        if self.feat_dir is not None:
            feat_path = self.feat_dir / f"{key}.npy"
            if feat_path.exists():
                feat = np.load(str(feat_path)).astype(np.float32)
                return self._correct_octave_shift(feat, key)

        return self._correct_octave_shift(
            self._extract_features_realtime(key), key
        )

    def _correct_octave_shift(self, feat: np.ndarray, key: str) -> np.ndarray:
        """
        检测 pyin 音高与 GT MIDI 之间的八度偏移并修正。

        当演唱者低/高八度哼唱时，pyin 提取的 midi_note 与 GT 标注
        相差约 12 个半音。通过比较中位音高自动检测并补偿。
        """
        import pretty_midi

        valid = feat[:, 1] > 0
        if valid.sum() < 10:
            return feat

        pyin_median = float(np.median(feat[valid, 0]))

        mid_path = self.midi_dir / f"{key}.mid"
        try:
            midi = pretty_midi.PrettyMIDI(str(mid_path))
            gt_pitches = [n.pitch for inst in midi.instruments
                          for n in inst.notes]
        except Exception:
            return feat

        if not gt_pitches:
            return feat

        gt_median = float(np.median(gt_pitches))
        raw_offset = pyin_median - gt_median

        # 将偏移量吸附到最近的12的倍数（只修正整八度偏移）
        octaves = round(raw_offset / 12)
        if octaves == 0:
            return feat

        correction = -octaves * 12
        feat = feat.copy()
        feat[valid, 0] += correction
        return feat

    def _extract_features_realtime(self, key: str) -> np.ndarray:
        """实时提取特征：优先 CREPE，不可用时回退到 librosa.pyin。"""
        import librosa

        wav_path = str(self.wav_dir / f"{key}.wav")
        audio, sr = librosa.load(wav_path, sr=16000, mono=True)

        try:
            import crepe
            time, freq, conf, _ = crepe.predict(
                audio, sr, step_size=FRAME_STEP_MS, viterbi=True
            )
        except ImportError:
            # 回退：librosa pYIN（已安装，无需 tensorflow）
            hop = int(sr * FRAME_STEP_S)
            freq, voiced_flag, voiced_prob = librosa.pyin(
                audio, sr=sr,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C6"),
                hop_length=hop,
                fill_na=0.0,
            )
            conf = voiced_prob.astype(np.float64)
            n = len(freq)
            time = np.arange(n) * FRAME_STEP_S

        midi_notes = 69.0 + 12.0 * np.log2(
            np.where(freq > 0, freq, 440.0) / 440.0
        )
        valid = (freq > 0).astype(np.float32)

        # beat position
        bpm, _ = librosa.beat.beat_track(y=audio, sr=sr)
        bpm = float(np.atleast_1d(bpm)[0])
        beat_dur = 60.0 / max(bpm, 1.0)
        beat_pos = (time % beat_dur) / beat_dur

        features = np.stack(
            [midi_notes.astype(np.float32),
             valid,
             conf.astype(np.float32),
             beat_pos.astype(np.float32)],
            axis=-1,
        )
        return features

    # ──────────────────────────────────────────────
    # 标注转换
    # ──────────────────────────────────────────────

    def _estimate_onset_offset(
        self, features: np.ndarray, midi: pretty_midi.PrettyMIDI, n_frames: int
    ) -> float:
        """
        用互相关估算 GT 时间与实际发声之间的全局偏移（秒）。

        方法：将 GT 音符活跃序列与特征的 voiced 序列做互相关，
        取相关最大处的 lag 作为偏移估计。

        Returns:
            float: offset_s，使得 actual_onset ≈ gt_onset + offset_s。
                   限制在 [-1.0, 1.0] 秒以防噪声影响。
        """
        from scipy.signal import correlate

        # GT 活跃序列：有音符的帧为 1，否则为 0
        gt_activity = np.zeros(n_frames, dtype=np.float32)
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                s = int(round(note.start / FRAME_STEP_S))
                e = int(round(note.end / FRAME_STEP_S))
                s = max(0, min(s, n_frames - 1))
                e = max(0, min(e, n_frames))
                if s < e:
                    gt_activity[s:e] = 1.0

        # 实际 voiced 序列：用置信度加权（CREPE 帧帧有值，纯 is_valid 无法区分噪声）
        voiced = (features[:n_frames, 1] * features[:n_frames, 2]).astype(np.float32)

        if gt_activity.sum() == 0 or voiced.sum() == 0:
            return 0.0

        # 互相关，找最大 lag
        corr = correlate(voiced, gt_activity, mode="full")
        lag = int(corr.argmax()) - (n_frames - 1)

        offset_s = lag * FRAME_STEP_S
        # 限制偏移范围，防止离群值
        offset_s = float(np.clip(offset_s, -1.0, 1.0))
        return offset_s

    def _load_labels(self, key: str, n_frames: int, features: np.ndarray | None = None) -> np.ndarray:
        """
        将 MIDI 音符转换为帧级 BIO 标签。

        若提供 features，则先估算 GT 与实际发声的时间偏移，
        将音符时间对齐到实际发声位置后再打标签。

        Args:
            key: 样本 ID。
            n_frames: 帧数（与特征对齐）。
            features: 帧级特征（可选），用于对齐偏移估算。

        Returns:
            np.ndarray: shape=(T,)，int64，取值 {0:O, 1:B, 2:I}。
        """
        mid_path = self.midi_dir / f"{key}.mid"
        midi = pretty_midi.PrettyMIDI(str(mid_path))
        labels = np.zeros(n_frames, dtype=np.int64)

        # 估算时间偏移
        offset_s = 0.0
        if features is not None:
            offset_s = self._estimate_onset_offset(features, midi, n_frames)

        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                # 将音符起止时间映射到帧索引（含偏移修正）
                start_frame = int(round((note.start + offset_s) / FRAME_STEP_S))
                end_frame = int(round((note.end + offset_s) / FRAME_STEP_S))
                start_frame = max(0, min(start_frame, n_frames - 1))
                end_frame = max(0, min(end_frame, n_frames))

                if start_frame >= end_frame:
                    continue
                labels[start_frame] = BIO_B
                if end_frame > start_frame + 1:
                    labels[start_frame + 1:end_frame] = BIO_I

        return labels


# ──────────────────────────────────────────────
# Collate：处理变长序列
# ──────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    """
    将不等长序列补零对齐为 batch tensor。

    Returns:
        dict:
            - keys (list[str])
            - features (Tensor): (B, T_max, 4)
            - labels (Tensor): (B, T_max)
            - mask (Tensor): (B, T_max) bool，有效帧为 True
            - n_frames (list[int])
    """
    keys = [item["key"] for item in batch]
    n_frames = [item["n_frames"] for item in batch]
    t_max = max(n_frames)

    feat_dim = batch[0]["features"].shape[-1]
    B = len(batch)

    features = torch.zeros(B, t_max, feat_dim, dtype=torch.float32)
    labels = torch.zeros(B, t_max, dtype=torch.int64)
    mask = torch.zeros(B, t_max, dtype=torch.bool)

    for i, item in enumerate(batch):
        T = item["n_frames"]
        features[i, :T] = item["features"]
        labels[i, :T] = item["labels"]
        mask[i, :T] = True

    return {
        "keys": keys,
        "features": features,
        "labels": labels,
        "mask": mask,
        "n_frames": n_frames,
    }
