"""
train/metrics.py - 量化评估指标

实现音符准确率（Note Accuracy）计算，
用于对比 RoundingBaselineQuantizer 与 BiLSTM-CRF 的性能。
"""

from dataclasses import dataclass

import numpy as np
import pretty_midi


# 音符匹配容差
ONSET_TOLERANCE = 0.30    # 起始时间容差：300ms（HumTrans GT 与实际发声有 200-400ms 系统偏移）
OFFSET_TOLERANCE = 0.05   # 结束时间容差：50ms（offset 评估时用）


@dataclass
class NoteMetrics:
    """音符评估结果。"""
    precision: float
    recall: float
    f1: float
    note_accuracy: float       # = recall（以GT为分母）
    n_pred: int
    n_gt: int
    n_matched: int

    def __str__(self) -> str:
        return (
            f"Precision={self.precision:.3f}  "
            f"Recall={self.recall:.3f}  "
            f"F1={self.f1:.3f}  "
            f"NoteAcc={self.note_accuracy:.3f}  "
            f"(pred={self.n_pred} gt={self.n_gt} matched={self.n_matched})"
        )


def _notes_from_midi(midi: pretty_midi.PrettyMIDI) -> list[tuple[int, float, float]]:
    """
    从 PrettyMIDI 提取音符列表。

    Returns:
        list of (pitch, onset, offset)
    """
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            notes.append((note.pitch, note.start, note.end))
    notes.sort(key=lambda x: x[1])
    return notes


def compute_note_metrics(
    pred_midi: pretty_midi.PrettyMIDI,
    gt_midi: pretty_midi.PrettyMIDI,
    onset_only: bool = True,
) -> NoteMetrics:
    """
    计算预测MIDI与标注MIDI之间的音符评估指标。

    匹配规则（onset_only=True 时）：
        一个预测音符被认为正确，当且仅当存在未匹配的GT音符满足：
        - pitch 完全相同
        - |onset_pred - onset_gt| <= ONSET_TOLERANCE (50ms)

    Args:
        pred_midi: 量化器输出的 PrettyMIDI。
        gt_midi: 标注的 PrettyMIDI（ground truth）。
        onset_only: 为 True 时只检查音高+起始时间；
                    为 False 时同时检查结束时间。

    Returns:
        NoteMetrics: 包含 precision / recall / f1 / note_accuracy。
    """
    pred_notes = _notes_from_midi(pred_midi)
    gt_notes = _notes_from_midi(gt_midi)

    if not gt_notes:
        if not pred_notes:
            return NoteMetrics(1.0, 1.0, 1.0, 1.0, 0, 0, 0)
        return NoteMetrics(0.0, 0.0, 0.0, 0.0, len(pred_notes), 0, 0)

    if not pred_notes:
        return NoteMetrics(0.0, 0.0, 0.0, 0.0, 0, len(gt_notes), 0)

    # 贪心匹配：对每个预测音符找最近的未匹配GT音符
    gt_matched = [False] * len(gt_notes)
    n_matched = 0

    for p_pitch, p_onset, p_offset in pred_notes:
        best_idx = -1
        best_dt = float("inf")

        for j, (g_pitch, g_onset, g_offset) in enumerate(gt_notes):
            if gt_matched[j]:
                continue
            if g_pitch != p_pitch:
                continue
            dt = abs(p_onset - g_onset)
            if dt > ONSET_TOLERANCE:
                continue
            if not onset_only:
                if abs(p_offset - g_offset) > OFFSET_TOLERANCE:
                    continue
            if dt < best_dt:
                best_dt = dt
                best_idx = j

        if best_idx >= 0:
            gt_matched[best_idx] = True
            n_matched += 1

    n_pred = len(pred_notes)
    n_gt = len(gt_notes)

    precision = n_matched / n_pred if n_pred > 0 else 0.0
    recall = n_matched / n_gt if n_gt > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return NoteMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        note_accuracy=recall,
        n_pred=n_pred,
        n_gt=n_gt,
        n_matched=n_matched,
    )


def evaluate_dataset(
    results: list[tuple[pretty_midi.PrettyMIDI, pretty_midi.PrettyMIDI]],
    onset_only: bool = True,
) -> NoteMetrics:
    """
    在整个数据集上计算平均指标（macro average）。

    Args:
        results: list of (pred_midi, gt_midi) 元组。
        onset_only: 是否只评估 onset。

    Returns:
        NoteMetrics: 数据集级别的平均指标。
    """
    if not results:
        return NoteMetrics(0.0, 0.0, 0.0, 0.0, 0, 0, 0)

    precisions, recalls, f1s, accs = [], [], [], []
    total_pred = total_gt = total_matched = 0

    for pred_midi, gt_midi in results:
        m = compute_note_metrics(pred_midi, gt_midi, onset_only=onset_only)
        precisions.append(m.precision)
        recalls.append(m.recall)
        f1s.append(m.f1)
        accs.append(m.note_accuracy)
        total_pred += m.n_pred
        total_gt += m.n_gt
        total_matched += m.n_matched

    return NoteMetrics(
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        f1=float(np.mean(f1s)),
        note_accuracy=float(np.mean(accs)),
        n_pred=total_pred,
        n_gt=total_gt,
        n_matched=total_matched,
    )
