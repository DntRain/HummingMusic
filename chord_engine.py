import librosa
import numpy as np
from music21 import pitch
import time


class MusicBrain:
    """
    HummingMusic 推理引擎 V5.2
    核心职责：高效执行时频分析、音高量化与启发式和弦映射。
    """

    def __init__(self):
        # 扩展和弦库（14种，支持4种调性）
        self.chord_library = {
            'C': 'C-E-G (C Major)', 'Am': 'A-C-E (A Minor)',
            'G': 'G-B-D (G Major)', 'Em': 'E-G-B (E Minor)',
            'F': 'F-A-C (F Major)', 'Dm': 'D-F-A (D Minor)',
            'D': 'D-F#-A (D Major)', 'Bm': 'B-D-F# (B Minor)',
            'Bb': 'Bb-D-F (Bb Major)', 'E': 'E-G#-B (E Major)',
            'A': 'A-C#-E (A Major)', 'Gm': 'G-Bb-D (G Minor)',
            'Fm': 'F-Ab-C (F Minor)', 'Ab': 'Ab-C-Eb (Ab Major)'
        }
        self.note_to_chord = {
            'C': 'C', 'D': 'G', 'E': 'C', 'F': 'F', 'G': 'C', 'A': 'F', 'B': 'G',
            'F#': 'D', 'C#': 'A', 'G#': 'E', 'Bb': 'Bb', 'Eb': 'Ab'
        }
        self._pitch_tool = pitch.Pitch()

    def predict(self, file_path):
        """生产级推理接口：专注于识别准确性与防御性编程"""
        try:
            # 采用 2s 黄金窗口与 16kHz 采样，平衡精度与速度
            y, sr = librosa.load(file_path, sr=16000, duration=2.0)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=1024)

            idx = magnitudes.argmax()
            freq = pitches[np.unravel_index(idx, magnitudes.shape)]

            if freq <= 0: return "N/A", "C-Major"

            self._pitch_tool.frequency = freq
            note_name = self._pitch_tool.name
            root_key = self.note_to_chord.get(note_name, 'C')

            return note_name, self.chord_library.get(root_key, 'C Major')
        except Exception:
            return "Error", "C-Major"