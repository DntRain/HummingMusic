"""
src/style_postprocess.py - 旋律 MIDI 的风格化后处理

输入：单轨主旋律 PrettyMIDI（decoder 输出 / 量化器输出均可）
输出：多轨风格化 PrettyMIDI（主旋律 + 伴奏，按风格做不同的节奏/和声变换）

风格规则：
    pop       block chord 伴奏（4 分音符）
    jazz      swing 化主旋律 + 七和弦琶音
    classical 长音颤音 + 强拍重音 + 阿尔贝蒂低音
    folk      交替低音（根音 / 五度）+ 简单 strum
"""

from __future__ import annotations

import copy
from typing import Callable

import pretty_midi

from src.style_transfer import STYLE_PROGRAMS


CHORD_PROGRESSIONS = {
    "pop": [0, 7, 9, 5],          # I  - V  - vi - IV  (C G Am F)
    "jazz": [0, 9, 2, 7],         # Imaj7 - vi7 - ii7 - V7
    "classical": [0, 5, 7, 0],    # I  - IV - V  - I
    "folk": [0, 7, 0, 5],         # I  - V  - I  - IV
}

CHORD_TYPE = {
    "pop": (0, 4, 7),             # 大三
    "jazz": (0, 4, 7, 10),        # 属七 / 小七（统一七和弦感）
    "classical": (0, 4, 7),
    "folk": (0, 7),               # 根 + 五度（power chord 感）
}


_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]


def _estimate_key_root(pm: pretty_midi.PrettyMIDI) -> int:
    """Krumhansl-Schmuckler 简化版：按音符时长加权 pc 直方图，与大调 profile 做相关。"""
    hist = [0.0] * 12
    for inst in pm.instruments:
        for n in inst.notes:
            hist[n.pitch % 12] += max(n.end - n.start, 0.01)
    total = sum(hist) or 1.0
    hist = [h / total for h in hist]
    best_root, best_score = 0, -1e9
    for root in range(12):
        score = sum(hist[(root + i) % 12] * _MAJOR_PROFILE[i] for i in range(12))
        if score > best_score:
            best_score, best_root = score, root
    return best_root


def _melody_inst(pm: pretty_midi.PrettyMIDI) -> pretty_midi.Instrument:
    return max(pm.instruments, key=lambda i: len(i.notes))


def _swing(notes: list[pretty_midi.Note], ratio: float = 2.0 / 3.0) -> None:
    """成对的等长短音符（< 0.4s）做 2:1 swing：第一个延长，第二个后推。"""
    i = 0
    while i + 1 < len(notes):
        a, b = notes[i], notes[i + 1]
        da, db = a.end - a.start, b.end - b.start
        if abs(da - db) < 0.05 and da < 0.4 and abs(b.start - a.end) < 0.02:
            total = da + db
            new_a_end = a.start + total * ratio
            a.end = new_a_end
            b.start = new_a_end
            i += 2
        else:
            i += 1


def _add_trill(notes: list[pretty_midi.Note], min_dur: float = 0.5,
               step: int = 1, rate: float = 0.0625) -> list[pretty_midi.Note]:
    """长音颤音：把 > min_dur 的音替换为本音/上邻音交替。"""
    out = []
    for n in notes:
        if n.end - n.start < min_dur:
            out.append(n)
            continue
        t = n.start
        toggle = 0
        while t < n.end:
            p = n.pitch + (step if toggle else 0)
            e = min(t + rate, n.end)
            out.append(pretty_midi.Note(velocity=n.velocity, pitch=p, start=t, end=e))
            t = e
            toggle ^= 1
    return out


def _beat_accent(notes: list[pretty_midi.Note], beat_dur: float,
                 strong: int = 115, weak: int = 80) -> None:
    """按拍位置加重音：每小节第 1、3 拍加重，2、4 减弱。"""
    for n in notes:
        beat = n.start / beat_dur
        on_strong = abs(beat - round(beat)) < 0.15 and int(round(beat)) % 2 == 0
        n.velocity = strong if on_strong else weak


def _apply_envelope(notes: list[pretty_midi.Note], beat_dur: float, style: str) -> None:
    """情感力度 envelope：给 velocity 加风格化的连续起伏曲线（不再二值跳变）。"""
    import math
    if not notes:
        return
    total = max(notes[-1].end, 1.0)

    for n in notes:
        beat = n.start / beat_dur
        beat_frac = beat - int(beat)
        rel = n.start / total                          # 0..1 整曲位置

        if style == "pop":
            # 4/4 强弱拍模板：拍 1 +12，拍 2 -6，拍 3 +8，拍 4 -6
            pat = [12, -6, 8, -6]
            base, delta = 92, pat[int(beat) % 4]
        elif style == "jazz":
            # 慵懒的 sin 飘动 + 弱拍稍重（swing 感）
            base = 88
            delta = int(8 * math.sin(2 * math.pi * beat / 4))
            if beat_frac > 0.5:                         # 反拍稍重
                delta += 4
        elif style == "classical":
            # 大幅 crescendo-decrescendo + 句末 ritardando 渐弱
            base = 70
            # 主轮廓：sin 曲线（峰值在中段）
            curve = int(35 * math.sin(math.pi * rel))
            # 句末渐弱（最后 20%）
            tail = -25 * max(0.0, rel - 0.8) / 0.2
            delta = curve + int(tail)
        elif style == "folk":
            # 平稳：小幅起伏 + 拍 1 微重
            base = 88
            delta = 6 if (abs(beat - round(beat)) < 0.15 and int(round(beat)) % 4 == 0) else 0
            delta += int(4 * math.sin(2 * math.pi * rel))
        else:
            base, delta = 90, 0

        # 高音稍重、低音稍轻（模拟人声共鸣）
        pitch_bias = (n.pitch - 60) // 6                # 每 6 半音 +/- 1
        n.velocity = max(30, min(120, base + delta + pitch_bias))


def _apply_articulation(notes: list[pretty_midi.Note], style: str,
                        gap_thresh: float = 0.12) -> None:
    """音长 articulation：jazz staccato；classical/folk legato；pop 中等。"""
    if not notes:
        return
    notes.sort(key=lambda n: n.start)
    ratio = {"pop": 0.88, "jazz": 0.62, "classical": 1.0, "folk": 0.98}.get(style, 0.9)

    for i, n in enumerate(notes):
        dur = n.end - n.start
        if style in ("classical", "folk"):
            # legato：若与下一音几乎相邻（间隙 < gap_thresh），连到下一音起点
            if i + 1 < len(notes) and notes[i + 1].start - n.end < gap_thresh:
                n.end = max(n.start + 0.04, notes[i + 1].start - 0.005)
            else:
                n.end = n.start + max(0.04, dur * ratio)
        else:
            n.end = n.start + max(0.04, dur * ratio)


def _make_chord_track(style: str, key_root: int, total: float, beat_dur: float,
                      program: int) -> pretty_midi.Instrument:
    """按 CHORD_PROGRESSIONS 生成伴奏轨；伴奏音量统一压低（避免盖过主旋律）。"""
    inst = pretty_midi.Instrument(program=program, name=f"{style}_chord")
    prog = CHORD_PROGRESSIONS[style]
    tones = CHORD_TYPE[style]
    # 和弦切换粒度：每 2 小节 1 次（8 拍），整段 9-10s ≈ 1-2 个和弦，不至于飘
    chord_dur = beat_dur * 8

    t = 0.0
    idx = 0
    while t < total:
        root = 48 + (key_root + prog[idx % len(prog)]) % 12  # C3 起点
        pitches = [root + off for off in tones]
        dur = min(chord_dur, total - t)

        if style == "jazz":
            # 慢琶音：每 2 拍弹一个音，整组持续到下个和弦
            step = beat_dur * 2
            for k in range(int(dur / step)):
                p = pitches[k % len(pitches)]
                inst.notes.append(pretty_midi.Note(
                    velocity=50, pitch=p, start=t + k * step,
                    end=min(t + (k + 1) * step + 0.3, t + dur)))
        elif style == "classical":
            # 阿尔贝蒂半速：8 分音符 → 2 拍一次
            pat = [pitches[0], pitches[-1], pitches[len(pitches) // 2], pitches[-1]]
            sub = beat_dur
            n_sub = int(dur / sub)
            for k in range(n_sub):
                p = pat[k % len(pat)]
                inst.notes.append(pretty_midi.Note(
                    velocity=45, pitch=p, start=t + k * sub,
                    end=t + (k + 1) * sub))
        else:  # pop / folk: 半音符 block chord（2 拍一击，留呼吸）
            sub = beat_dur * 2
            n_sub = int(dur / sub)
            for k in range(n_sub):
                vel = 52 if k % 2 == 0 else 42
                for p in pitches:
                    inst.notes.append(pretty_midi.Note(
                        velocity=vel, pitch=p,
                        start=t + k * sub,
                        end=t + (k + 1) * sub - 0.02))

        t += dur
        idx += 1
    return inst


def _make_bass_track(style: str, key_root: int, total: float, beat_dur: float,
                     program: int) -> pretty_midi.Instrument:
    """低音轨：和弦切换与 chord 轨同步（8 拍/和弦）；音量适中，不抢主旋律。"""
    inst = pretty_midi.Instrument(program=program, name=f"{style}_bass")
    prog = CHORD_PROGRESSIONS[style]
    t = 0.0
    idx = 0
    chord_dur = beat_dur * 8
    while t < total:
        root = 36 + (key_root + prog[idx % len(prog)]) % 12  # C2 起点
        fifth = root + 7
        dur = min(chord_dur, total - t)
        n_beats = max(int(dur / beat_dur), 1)
        for k in range(n_beats):
            if style == "folk":
                p = root if k % 2 == 0 else fifth
            elif style == "jazz":
                p = [root, root + 4, fifth, fifth + 2][k % 4]   # walking
            elif style == "classical":
                # 古典低音半音符长音，只在 1、3 拍换
                if k % 2 != 0:
                    continue
                p = root if (k // 2) % 2 == 0 else fifth
            else:  # pop: 1 拍 root，3 拍 fifth
                if k % 2 != 0:
                    continue
                p = root if (k // 2) % 2 == 0 else fifth
            note_end = t + (k + 2) * beat_dur if style in ("classical", "pop") else t + (k + 1) * beat_dur
            note_end = min(note_end - 0.02, t + dur)
            inst.notes.append(pretty_midi.Note(
                velocity=58, pitch=p, start=t + k * beat_dur, end=note_end))
        t += dur
        idx += 1
    return inst


def _quantize_notes(notes: list[pretty_midi.Note], beat_dur: float,
                    grid: int = 2) -> None:
    """把 note 的 start/end 量化到 1/grid 拍网格（默认 8 分音符，更稳）。"""
    step = beat_dur / grid
    for n in notes:
        n.start = round(n.start / step) * step
        n.end = round(n.end / step) * step
        if n.end <= n.start:
            n.end = n.start + step


def _shift_to_zero(notes: list[pretty_midi.Note]) -> None:
    """平移让最早音的 start = 0，让旋律与伴奏的拍 1 对齐。"""
    if not notes:
        return
    offset = min(n.start for n in notes)
    if offset == 0:
        return
    for n in notes:
        n.start -= offset
        n.end -= offset


def _make_drum_track(style: str, total: float, beat_dur: float) -> pretty_midi.Instrument:
    """GM 鼓组：BD=36, SD=38, CH=42, OH=46, Ride=51, Tamb=54."""
    inst = pretty_midi.Instrument(program=0, is_drum=True, name=f"{style}_drums")
    bar = beat_dur * 4

    def add(p, t, v=90, dur=0.08):
        if t < total:
            inst.notes.append(pretty_midi.Note(velocity=v, pitch=p, start=t, end=min(t + dur, total)))

    n_bars = int(total / bar) + 1
    for b in range(n_bars):
        t0 = b * bar
        if style == "pop":
            # 标准 4/4：BD 1+3，SD 2+4，CH 每 8 分音符
            add(36, t0 + 0 * beat_dur, 105)
            add(38, t0 + 1 * beat_dur, 95)
            add(36, t0 + 2 * beat_dur, 100)
            add(38, t0 + 3 * beat_dur, 95)
            for k in range(8):
                add(42, t0 + k * beat_dur / 2, 70)
        elif style == "jazz":
            # Swing ride：1, 2+swing 8, 3, 4+swing 8；snare 2+4 ghost；BD 1+3 弱
            for k in range(4):
                add(51, t0 + k * beat_dur, 85)        # ride downbeat
                add(51, t0 + (k + 2 / 3) * beat_dur, 70)  # swing eighth
            add(38, t0 + 1 * beat_dur, 55)            # ghost snare
            add(38, t0 + 3 * beat_dur, 55)
            add(36, t0 + 0 * beat_dur, 60)
            add(36, t0 + 2 * beat_dur, 55)
        elif style == "folk":
            # 简单 tambourine 4 分音符 + 偶尔 BD
            for k in range(4):
                v = 80 if k % 2 == 0 else 65
                add(54, t0 + k * beat_dur, v)
            add(36, t0 + 0 * beat_dur, 75)
            add(36, t0 + 2 * beat_dur, 70)
        # classical: 不加鼓
    return inst


def _stylize_pop(ns, bd):
    _shift_to_zero(ns)
    _quantize_notes(ns, bd)
    _apply_envelope(ns, bd, "pop")
    _apply_articulation(ns, "pop")
    return ns


def _stylize_jazz(ns, bd):
    _shift_to_zero(ns)
    _quantize_notes(ns, bd)
    _swing(ns)
    _apply_envelope(ns, bd, "jazz")
    _apply_articulation(ns, "jazz")
    return ns


def _stylize_classical(ns, bd):
    _shift_to_zero(ns)
    _quantize_notes(ns, bd)
    _apply_envelope(ns, bd, "classical")
    _apply_articulation(ns, "classical")
    return _add_trill(ns)


def _stylize_folk(ns, bd):
    _shift_to_zero(ns)
    _quantize_notes(ns, bd)
    _apply_envelope(ns, bd, "folk")
    _apply_articulation(ns, "folk")
    return ns


STYLIZERS: dict[str, Callable[[list[pretty_midi.Note], float], list[pretty_midi.Note]]] = {
    "pop": _stylize_pop,
    "jazz": _stylize_jazz,
    "classical": _stylize_classical,
    "folk": _stylize_folk,
}


def stylize(pm: pretty_midi.PrettyMIDI, style: str,
            tempo: float | None = None) -> pretty_midi.PrettyMIDI:
    """主入口：对主旋律 PrettyMIDI 套用风格规则，返回多轨 PrettyMIDI。"""
    if style not in CHORD_PROGRESSIONS:
        raise ValueError(f"unknown style: {style}")

    if tempo is None:
        try:
            tempo = float(pm.estimate_tempo())
        except Exception:
            tempo = 120.0
    if not (40 < tempo < 240):
        tempo = 120.0
    beat_dur = 60.0 / tempo

    out = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    progs = STYLE_PROGRAMS[style]

    src = _melody_inst(pm)
    notes = [copy.deepcopy(n) for n in sorted(src.notes, key=lambda n: n.start)]
    notes = STYLIZERS[style](notes, beat_dur)

    melody = pretty_midi.Instrument(program=progs["melody"], name="melody")
    melody.notes.extend(notes)
    out.instruments.append(melody)

    total = max(out.get_end_time(), pm.get_end_time(), 1.0)
    key = _estimate_key_root(pm)

    out.instruments.append(_make_chord_track(style, key, total, beat_dur, progs["chords"]))
    out.instruments.append(_make_bass_track(style, key, total, beat_dur, progs["bass"]))
    if style != "classical":
        out.instruments.append(_make_drum_track(style, total, beat_dur))
    return out
