"""
test_regression_week14.py - Week14 回归测试用例集（≥20 例）

目的：协助 bug 修复验证 + 锁定已修复缺陷不再回归 + 把当前已知缺陷
（Bug-A/Bug-B，真实哼唱产出 0 notes）以 xfail 形式编码进测试套。

用例与缺陷映射见 reports/week14/ly/regression_test_log.md（文档矩阵）。

设计约束：全部 CI-safe——
- 纯逻辑用例不依赖 torch / 模型 / 真实音频，CI（mock torch、无 .pt）可跑；
- 需要真实 torchcrepe + 模型的端到端用例用 skipif(文件缺失) + xfail 标注。
"""
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio_processing import (  # noqa: E402
    LowEnergyError,
    _check_rms_energy,
    _estimate_bpm,
    _filter_low_confidence,
)
from src.quantizer import (  # noqa: E402
    BIO_TAGS,
    RoundingBaselineQuantizer,
    _bio_to_notes,
)
from src import quantizer as _q  # noqa: E402
from src import style_transfer as _st  # noqa: E402
from src.style_transfer import decode_recon_to_notes, transfer_style  # noqa: E402
from src.style_postprocess import stylize  # noqa: E402

REAL_AUDIO = ROOT / "example_1.m4a"


def _pitch_data(duration=2.0, freq=440.0, conf=0.95, bpm=120.0, step=0.01):
    t = np.arange(0, duration, step)
    n = len(t)
    return {
        "time": t,
        "frequency": np.full(n, freq, dtype=np.float64),
        "confidence": np.full(n, conf, dtype=np.float64),
        "bpm": bpm,
    }


# ══════════════════════════════════════════════════════════
# RC-01~04  Bug-04 三层短路（RMS 门控 / 有效帧 / 空 MIDI）
# ══════════════════════════════════════════════════════════

class TestRC_Bug04ShortCircuit:
    def test_rc01_rms_digital_silence_blocked(self):
        """RC-01: 全零数字静音被 RMS 门拦截。"""
        with pytest.raises(LowEnergyError):
            _check_rms_energy(np.zeros(16000, dtype=np.float32))

    def test_rc02_rms_low_noise_blocked(self):
        """RC-02: -60 dBFS 底噪被拦截。"""
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(16000).astype(np.float32) * 1e-3
        with pytest.raises(LowEnergyError):
            _check_rms_energy(audio)

    def test_rc03_rms_normal_humming_passes(self):
        """RC-03: 正常哼唱（-23 dBFS）放行。"""
        t = np.arange(16000) / 16000
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        _check_rms_energy(audio)  # 不抛即通过

    def test_rc04_low_valid_ratio_empty_no_model(self, monkeypatch):
        """RC-04: 有效帧 <10% 时短路返回空 MIDI，且不加载模型。"""
        calls = {"n": 0}
        monkeypatch.setattr(_q, "_load_model",
                            lambda: calls.__setitem__("n", calls["n"] + 1))
        pd = _pitch_data(duration=10.0)
        pd["frequency"][:] = np.nan  # 0% 有效
        pd["confidence"][:] = 0.0
        midi = _q.quantize_humming(pd)
        assert sum(len(i.notes) for i in midi.instruments) == 0
        assert calls["n"] == 0


# ══════════════════════════════════════════════════════════
# RC-05~07  BPM=0 零除兜底（验收期发现的 Bug-A2）
# ══════════════════════════════════════════════════════════

class TestRC_BpmFallback:
    def test_rc05_pure_tone_no_zero_div(self):
        """RC-05: 纯音 librosa 可能返回 tempo=0，应回退 120 不崩。"""
        t = np.arange(16000) / 16000
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        bpm = _estimate_bpm(audio, 16000)
        assert np.isfinite(bpm) and bpm >= 30.0

    def test_rc06_silence_no_zero_div(self):
        """RC-06: 极短/近静音输入不触发零除。"""
        bpm = _estimate_bpm(np.zeros(800, dtype=np.float32), 16000)
        assert bpm == 120.0

    def test_rc07_prettymidi_init_bpm_safe(self):
        """RC-07: 兜底后的 BPM 喂给 PrettyMIDI 不抛 ZeroDivisionError。"""
        pretty_midi.PrettyMIDI(initial_tempo=_estimate_bpm(
            np.zeros(800, dtype=np.float32), 16000))


# ══════════════════════════════════════════════════════════
# RC-08~12  Bug-B 根因：BIO 解码契约（无 B 则无 note）
#           纯逻辑，不依赖模型，直接锁定退化模型的失效机理
# ══════════════════════════════════════════════════════════

class TestRC_BioDecodingContract:
    @staticmethod
    def _notes(tag_names, pitch=60.0):
        idx = [BIO_TAGS[x] for x in tag_names]
        n = len(idx)
        t = np.arange(n) * 0.1
        midi = np.full(n, pitch, dtype=np.float64)
        return _bio_to_notes(idx, t, midi)

    def test_rc08_all_I_no_B_yields_zero(self):
        """RC-08: 全 I 无 B（退化模型输出）→ 0 note。复现 Bug-B 机理。"""
        assert len(self._notes(["I"] * 8)) == 0

    def test_rc09_all_O_yields_zero(self):
        """RC-09: 全 O → 0 note。"""
        assert len(self._notes(["O"] * 8)) == 0

    def test_rc10_BII_yields_one(self):
        """RC-10: B-I-I → 1 note。"""
        assert len(self._notes(["B", "I", "I", "I"])) == 1

    def test_rc11_three_B_yields_three(self):
        """RC-11: B-B-B-B → 4 note（每个 onset 起一个）。"""
        assert len(self._notes(["B", "B", "B", "B"])) == 4

    def test_rc12_B_O_B_yields_two(self):
        """RC-12: B-I-O-B-I → 2 note（O 收束前一个）。"""
        assert len(self._notes(["B", "I", "O", "B", "I"])) == 2


# ══════════════════════════════════════════════════════════
# RC-13~15  Bug-A 根因：置信度过滤机理
# ══════════════════════════════════════════════════════════

class TestRC_ConfidenceFilter:
    def test_rc13_below_threshold_to_nan(self):
        """RC-13: 低于阈值的帧频率被置 NaN。"""
        freq = np.array([440.0, 440.0, 440.0])
        conf = np.array([0.05, 0.05, 0.05])  # 远低于任何合理阈值
        out = _filter_low_confidence(freq, conf)
        assert np.isnan(out).all()

    def test_rc14_above_threshold_kept(self):
        """RC-14: 高置信度帧保留。"""
        freq = np.array([440.0, 440.0])
        conf = np.array([0.99, 0.99])
        out = _filter_low_confidence(freq, conf)
        assert np.allclose(out, 440.0)

    def test_rc15_all_low_conf_all_nan(self):
        """RC-15: 全部低置信度 → 全 NaN（Bug-A 在 viterbi 下的失效模式）。"""
        freq = np.full(100, 440.0)
        conf = np.full(100, 0.001)  # 模拟 viterbi periodicity 塌缩
        out = _filter_low_confidence(freq, conf)
        assert np.isnan(out).all()


# ══════════════════════════════════════════════════════════
# RC-16~18  baseline 量化器（Bug-B 当前兜底路径，须正常产 note）
# ══════════════════════════════════════════════════════════

class TestRC_BaselineQuantizer:
    def test_rc16_baseline_produces_notes(self):
        """RC-16: baseline 对有效输入产出 >0 note。"""
        midi = RoundingBaselineQuantizer()(_pitch_data(duration=2.0))
        assert sum(len(i.notes) for i in midi.instruments) > 0

    def test_rc17_baseline_velocity_80(self):
        """RC-17: baseline 力度固定 80。"""
        midi = RoundingBaselineQuantizer()(_pitch_data(duration=2.0))
        assert midi.instruments[0].notes[0].velocity == 80

    def test_rc18_baseline_deterministic(self):
        """RC-18: 同输入两次产出完全一致。"""
        pd = _pitch_data(duration=2.0)

        def sig(m):
            return [(round(n.start, 4), round(n.end, 4), n.pitch, n.velocity)
                    for i in m.instruments for n in i.notes]
        assert sig(RoundingBaselineQuantizer()(pd)) == \
            sig(RoundingBaselineQuantizer()(pd))


# ══════════════════════════════════════════════════════════
# RC-19~23  风格迁移解码 / 后处理 / 短路
# ══════════════════════════════════════════════════════════

class TestRC_StyleTransfer:
    def test_rc19_decode_empty_returns_empty(self):
        """RC-19: 空概率图 → 空 note 列表。"""
        assert decode_recon_to_notes(
            np.zeros((10, 0)), 48, 84, 0.05) == []

    def test_rc20_decode_single_region_one_note(self):
        """RC-20: 单个连续高概率区间 → 1 note。"""
        prob = np.zeros((12, 10))
        prob[3, 2:8] = 0.9  # 相对 pitch=3，第 2~7 帧激活
        notes = decode_recon_to_notes(prob, 48, 60, 0.05,
                                      prob_thresh=0.2, min_note_dur=0.06)
        assert len(notes) == 1
        assert notes[0].pitch == 51

    def test_rc21_decode_min_duration_filters(self):
        """RC-21: 短于 min_note_dur 的单帧噪声被丢弃。"""
        prob = np.zeros((12, 10))
        prob[3, 4] = 0.9  # 仅 1 帧 = 0.05s < 0.06s
        notes = decode_recon_to_notes(prob, 48, 60, 0.05,
                                      prob_thresh=0.2, min_note_dur=0.06)
        assert notes == []

    def test_rc22_transfer_empty_midi_short_circuit(self, monkeypatch):
        """RC-22: 空 MIDI 不触发 VQ-VAE 加载，输出 0 note。"""
        calls = {"vq": 0}
        monkeypatch.setattr(_st, "_load_vqvae_model",
                            lambda: calls.__setitem__("vq", 1))
        monkeypatch.setattr(_st, "_load_style_decoders", lambda: {})
        monkeypatch.setattr(_st, "_load_style_vectors", lambda: {})
        out = transfer_style(pretty_midi.PrettyMIDI(), "pop")
        assert sum(len(i.notes) for i in out.instruments) == 0
        assert calls["vq"] == 0

    @pytest.mark.parametrize("style", ["pop", "jazz", "classical", "folk"])
    def test_rc23_stylize_multitrack(self, style):
        """RC-23: stylize 对单轨旋律产出多轨（旋律+伴奏）。"""
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=0, name="melody")
        for k in range(4):
            inst.notes.append(pretty_midi.Note(
                velocity=90, pitch=60 + k, start=k * 0.5, end=k * 0.5 + 0.4))
        pm.instruments.append(inst)
        out = stylize(pm, style, tempo=120.0)
        assert len(out.instruments) >= 2
        assert sum(len(i.notes) for i in out.instruments) > 0


# ══════════════════════════════════════════════════════════
# RC-24  端到端已知缺陷（Bug-A + Bug-B）：真实哼唱产出 0 notes
#         skipif：CI 无 m4a/torch；xfail：本地复现已知失败
# ══════════════════════════════════════════════════════════

@pytest.mark.skipif(not REAL_AUDIO.exists(),
                    reason="CI 无真实音频样本；本地复现用")
@pytest.mark.xfail(reason="Bug-A(viterbi periodicity 塌缩)+Bug-B(模型不出 B 标签)"
                          "导致真实哼唱 quantize 产出 0 notes，待修",
                   strict=False)
def test_rc24_real_humming_produces_notes():
    """RC-24: 真实哼唱端到端应产出 note（当前因 Bug-A/B 失败 → xfail）。"""
    from src.audio_processing import extract_pitch
    from src.quantizer import quantize_humming
    pd = extract_pitch(str(REAL_AUDIO))
    midi = quantize_humming(pd)
    assert sum(len(i.notes) for i in midi.instruments) > 0
