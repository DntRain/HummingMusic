"""
tools/gen_ppt.py - 生成中期答辩 PPT（全组版）
用法：python tools/gen_ppt.py
输出：reports/midterm_group.pptx

成员分工：
  徐亦轲 — 音频处理(CREPE F0) · 容错量化(BiLSTM-CRF)
  窦  畅 — 系统界面(Flask) · 风格推断(music21) · 渲染(FluidSynth)
  钟  翔 — 数据管道(DataLoader) · 边界防御量化器 · 底层合成
  李  炎 — BIO标注 · 主观评分 · 量化压缩对比
"""

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 注册微软雅黑用于图表中文
_MSYH = "/usr/share/fonts/WindowsFonts/msyh.ttc"
_MSYH_BD = "/usr/share/fonts/WindowsFonts/msyhbd.ttc"
for _fp in [_MSYH, _MSYH_BD]:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── 颜色 ──────────────────────────────────────
C_DARK   = RGBColor(0x1F, 0x38, 0x64)   # 深蓝
C_MID    = RGBColor(0x2E, 0x75, 0xB6)   # 中蓝
C_ACCENT = RGBColor(0xED, 0x7D, 0x31)   # 橙色
C_GREEN  = RGBColor(0x70, 0xAD, 0x47)   # 绿色（达标）
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT  = RGBColor(0xD6, 0xE4, 0xF0)   # 浅蓝背景
C_GRAY   = RGBColor(0x59, 0x59, 0x59)
C_BG     = RGBColor(0xF5, 0xF8, 0xFC)   # 页面背景
C_PURPLE = RGBColor(0x76, 0x30, 0x9A)   # 紫色（李炎）

FONT = "微软雅黑"
W, H = Inches(13.33), Inches(7.5)   # 16:9


# ── 基础工具函数 ───────────────────────────────

def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]   # 完全空白
    return prs.slides.add_slide(layout)


def add_rect(slide, l, t, w, h, fill=None, line=None, line_w=Pt(1)):
    shape = slide.shapes.add_shape(1, l, t, w, h)
    shape.line.fill.background()
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
        shape.line.width = line_w
    else:
        shape.line.fill.background()
    return shape


def add_text(slide, text, l, t, w, h,
             size=Pt(18), bold=False, color=None,
             align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(l, t, w, h)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color
    return txBox


def add_para(tf, text, size=Pt(16), bold=False, color=None,
             align=PP_ALIGN.LEFT, indent=0, italic=False, space_before=Pt(4)):
    p = tf.add_paragraph()
    p.alignment = align
    p.space_before = space_before
    if indent:
        p.level = indent
    run = p.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color
    return p


def slide_bg(slide, color=C_BG):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def header_bar(slide, title, subtitle=None):
    """顶部深蓝色标题栏"""
    add_rect(slide, Inches(0), Inches(0), W, Inches(1.1), fill=C_DARK)
    add_text(slide, title,
             Inches(0.4), Inches(0.1), Inches(10), Inches(0.85),
             size=Pt(28), bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text(slide, subtitle,
                 Inches(10.5), Inches(0.28), Inches(2.5), Inches(0.6),
                 size=Pt(14), color=C_LIGHT, align=PP_ALIGN.RIGHT)


def member_tag(slide, name, x, y, color=C_MID):
    """右上角成员标签"""
    add_rect(slide, x, y, Inches(1.5), Inches(0.32), fill=color)
    add_text(slide, f"  {name}",
             x, y, Inches(1.5), Inches(0.32),
             size=Pt(11), bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)


def card(slide, x, y, w, h, title, items, accent=C_MID, title_size=Pt(15)):
    """带左色条的卡片"""
    add_rect(slide, x, y, w, h, fill=C_WHITE, line=C_LIGHT, line_w=Pt(0.8))
    add_rect(slide, x, y, Inches(0.08), h, fill=accent)
    add_text(slide, title,
             x + Inches(0.18), y + Inches(0.07), w - Inches(0.25), Inches(0.4),
             size=title_size, bold=True, color=C_DARK)
    txBox = slide.shapes.add_textbox(
        x + Inches(0.18), y + Inches(0.5), w - Inches(0.28), h - Inches(0.6))
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(3)
        run = p.add_run()
        run.text = item
        run.font.name = FONT
        run.font.size = Pt(12)
        run.font.color.rgb = C_GRAY


# ════════════════════════════════════════════════════════
# Slide 1 — 封面（全组）
# ════════════════════════════════════════════════════════
def slide_cover(prs):
    slide = blank_slide(prs)
    slide_bg(slide, C_DARK)

    add_rect(slide, Inches(0), Inches(4.8), W, Inches(2.7), fill=C_MID)
    add_rect(slide, Inches(0), Inches(6.5), W, Inches(1.0), fill=C_ACCENT)

    add_text(slide, "基于人声哼唱的旋律转译与风格迁移系统",
             Inches(0.8), Inches(1.4), Inches(11.7), Inches(1.4),
             size=Pt(36), bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

    add_text(slide, "中  期  答  辩",
             Inches(0.8), Inches(2.9), Inches(11.7), Inches(0.8),
             size=Pt(26), color=C_LIGHT, align=PP_ALIGN.CENTER, italic=True)

    add_text(slide, "2026 年 4 月 20 日",
             Inches(0.8), Inches(5.0), Inches(11.7), Inches(0.55),
             size=Pt(18), color=C_WHITE, align=PP_ALIGN.CENTER)

    members = [
        ("徐亦轲", "音频处理 · 容错量化"),
        ("窦  畅",  "系统界面 · 风格推断 · 渲染"),
        ("钟  翔",  "数据管道 · 边界防御 · 合成"),
        ("李  炎",  "BIO标注 · 评测 · 压缩对比"),
    ]
    col_w = Inches(3.1)
    total = len(members) * float(col_w)
    start_x = (float(W) - total) / 2
    for i, (name, role) in enumerate(members):
        cx = start_x + i * float(col_w)
        add_text(slide, name,
                 cx, Inches(5.65), col_w, Inches(0.42),
                 size=Pt(16), bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        add_text(slide, role,
                 cx, Inches(6.1), col_w, Inches(0.35),
                 size=Pt(11), color=C_LIGHT, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════
# Slide 2 — 系统架构
# ════════════════════════════════════════════════════════
def slide_arch(prs):
    slide = blank_slide(prs)
    slide_bg(slide)
    header_bar(slide, "系统架构", "System Overview")

    modules = [
        ("哼唱输入\nWAV / MP3",  C_GRAY,   False, ""),
        ("音频处理\nCREPE F0",   C_MID,    True,  "徐亦轲"),
        ("容错量化\nBiLSTM-CRF", C_ACCENT, True,  "徐亦轲·钟翔"),
        ("风格推断\nmusic21",    C_MID,    True,  "窦畅"),
        ("音频渲染\nFluidSynth", C_MID,    True,  "窦畅·钟翔"),
    ]

    box_w = Inches(2.1)
    box_h = Inches(1.4)
    gap   = Inches(0.18)
    total = len(modules) * float(box_w) + (len(modules) - 1) * float(gap)
    start_x = (float(W) - total) / 2
    y = Inches(2.0)

    for i, (label, color, highlight, owner) in enumerate(modules):
        x = start_x + i * (float(box_w) + float(gap))
        if highlight:
            add_rect(slide, x + Inches(0.04), y + Inches(0.04),
                     box_w, box_h, fill=RGBColor(0x1A, 0x50, 0x88))
        add_rect(slide, x, y, box_w, box_h, fill=color)
        add_text(slide, label,
                 x, y + Inches(0.18), box_w, box_h,
                 size=Pt(15), bold=highlight, color=C_WHITE,
                 align=PP_ALIGN.CENTER)
        if owner:
            add_text(slide, owner,
                     x, y + box_h + Inches(0.05), box_w, Inches(0.3),
                     size=Pt(10), color=C_GRAY, align=PP_ALIGN.CENTER)
        if i < len(modules) - 1:
            ax = x + float(box_w) + Inches(0.02)
            ay = y + float(box_h) / 2 - Inches(0.12)
            add_text(slide, "▶", ax, ay, gap, Inches(0.3),
                     size=Pt(14), color=C_MID, align=PP_ALIGN.CENTER)

    io_labels = [
        (start_x, "WAV 录音"),
        (start_x + float(box_w) + float(gap), "time / freq\nconfidence / bpm"),
        (start_x + 2*(float(box_w)+float(gap)), "PrettyMIDI\n单轨旋律"),
        (start_x + 3*(float(box_w)+float(gap)), "PrettyMIDI\n旋律+和弦"),
        (start_x + 4*(float(box_w)+float(gap)), "WAV 音频"),
    ]
    for x, label in io_labels:
        add_text(slide, label, x, y + box_h + Inches(0.45),
                 box_w, Inches(0.6),
                 size=Pt(10), color=C_GRAY, align=PP_ALIGN.CENTER)

    # Gradio界面标注
    add_rect(slide, Inches(0.35), Inches(5.5), Inches(12.6), Inches(0.5),
             fill=RGBColor(0xE8, 0xF0, 0xFB), line=C_MID, line_w=Pt(1.2))
    add_text(slide, "  Gradio / Flask Web 界面（窦畅）— 全流程统一入口，Fetch API 异步，零刷新体验",
             Inches(0.45), Inches(5.52), Inches(12.2), Inches(0.45),
             size=Pt(12), color=C_DARK)

    add_rect(slide, Inches(0.35), Inches(6.1), Inches(6.0), Inches(0.55),
             fill=RGBColor(0xFD, 0xF0, 0xE6), line=C_ACCENT, line_w=Pt(1.5))
    add_text(slide, "  ■ 蓝/橙色为本报告重点模块",
             Inches(0.45), Inches(6.15), Inches(5.8), Inches(0.45),
             size=Pt(12), color=C_ACCENT)

    add_text(slide, "支持风格：Pop · Jazz · Classical · Folk",
             Inches(7.0), Inches(6.15), Inches(6.0), Inches(0.45),
             size=Pt(12), color=C_GRAY, align=PP_ALIGN.RIGHT)


# ════════════════════════════════════════════════════════
# Slide 3 — 数据工程（钟翔 + 李炎）
# ════════════════════════════════════════════════════════
def slide_data_engineering(prs):
    slide = blank_slide(prs)
    slide_bg(slide)
    header_bar(slide, "数据工程：管道构建与标注质控", "Data Engineering")
    member_tag(slide, "钟翔", Inches(9.0), Inches(0.15), C_GREEN)
    member_tag(slide, "李炎", Inches(10.7), Inches(0.15), C_PURPLE)

    # ── 左：钟翔 DataLoader ──────────────────────
    lx = Inches(0.4)
    add_rect(slide, lx, Inches(1.25), Inches(6.0), Inches(0.48), fill=C_GREEN)
    add_text(slide, "  数据管道  DataLoader（钟翔）",
             lx, Inches(1.25), Inches(6.0), Inches(0.48),
             size=Pt(16), bold=True, color=C_WHITE)

    zhong_cards = [
        ("动态 Padding DataLoader",
         ["Batch_size = 32，支持变长序列", "collate_fn 动态补零，保留真实时序边界"]),
        ("数据增强",
         ["音高偏移 ±2 半音 → 旋律调性泛化",
          "时间拉伸 ±10% → 节奏速度适应",
          "样本量 2,000 → 6,414 条  (+221%)"]),
        ("降采样优化",
         ["原始 44.1 kHz → 强制降至 22.05 kHz",
          "FFT 矩阵维度减半，运算加速 ~2×"]),
    ]
    for i, (t, items) in enumerate(zhong_cards):
        cy = Inches(1.85) + i * Inches(1.7)
        card(slide, lx, cy, Inches(6.0), Inches(1.55), t, items, C_GREEN)

    # ── 右：李炎 BIO 标注 ─────────────────────────
    rx = Inches(7.0)
    add_rect(slide, rx, Inches(1.25), Inches(5.9), Inches(0.48), fill=C_PURPLE)
    add_text(slide, "  BIO 标注质量控制（李炎）",
             rx, Inches(1.25), Inches(5.9), Inches(0.48),
             size=Pt(16), bold=True, color=C_WHITE)

    li_cards = [
        ("人工逐帧标注",
         ["样本量：2,000 条（HumTrans 训练集）",
          "三类标签：B-Note / I-Note / O",
          "100 条双人交叉验证"]),
        ("一致性验证",
         ["指标：Cohen's Kappa 系数（scikit-learn）",
          "平均 Kappa = 0.9618",
          "Kappa > 0.8 占比：100%",
          "一致性等级：优秀（几乎完全一致）"]),
    ]
    for i, (t, items) in enumerate(li_cards):
        cy = Inches(1.85) + i * Inches(2.5)
        card(slide, rx, cy, Inches(5.9), Inches(2.3), t, items, C_PURPLE)

    # 底部数据流向
    add_rect(slide, Inches(0.4), Inches(6.55), Inches(12.5), Inches(0.62), fill=C_DARK)
    add_text(slide, "  6,414 条增强样本  +  2,000 条高质量 BIO 标注  →  为 BiLSTM-CRF 训练提供数据底座",
             Inches(0.5), Inches(6.58), Inches(12.2), Inches(0.55),
             size=Pt(15), bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════
# Slide 4 — 音频处理（徐亦轲）
# ════════════════════════════════════════════════════════
def slide_audio_processing(prs):
    slide = blank_slide(prs)
    slide_bg(slide)
    header_bar(slide, "音频处理：CREPE F0 提取管线", "Audio Processing")
    member_tag(slide, "徐亦轲", Inches(9.0), Inches(0.15), C_MID)

    # ── 左：处理流程 ──────────────────────────────
    lx = Inches(0.4)
    steps = [
        ("① 加载与重采样",  C_MID,    "librosa.load → 单声道 16 kHz"),
        ("② CREPE 音高检测", C_ACCENT, "model=full，Viterbi 平滑\nstep=10ms，置信度阈值=0.8"),
        ("③ 低置信度过滤",   C_MID,    "conf < 0.8 → freq = NaN\n（对应哼唱间隙/静音段）"),
        ("④ 短静音插值",     C_MID,    "NaN 段时长 < 0.2s → 线性插值\n长静音段保留，作为音符边界"),
        ("⑤ BPM 估计",      C_GREEN,  "librosa.beat.beat_track\n输出 bpm 供量化器使用"),
    ]
    bw, bh, bgap = Inches(5.6), Inches(0.78), Inches(0.15)
    ly = Inches(1.3)
    for i, (label, color, desc) in enumerate(steps):
        by = ly + i * (bh + bgap)
        add_rect(slide, lx, by, bw, bh, fill=color)
        add_text(slide, label,
                 lx + Inches(0.15), by + Inches(0.04), bw, Inches(0.38),
                 size=Pt(14), bold=True, color=C_WHITE)
        add_text(slide, desc,
                 lx + Inches(0.15), by + Inches(0.38), bw - Inches(0.2), Inches(0.38),
                 size=Pt(11), color=C_WHITE)
        if i < len(steps) - 1:
            add_text(slide, "↓",
                     lx + bw / 2 - Inches(0.15), by + bh,
                     Inches(0.3), bgap + Inches(0.04),
                     size=Pt(13), color=C_MID, align=PP_ALIGN.CENTER)

    # ── 右：输出接口 + 关键参数 ────────────────────
    rx = Inches(6.5)
    add_text(slide, "输出接口（→ 量化器）",
             rx, Inches(1.3), Inches(6.5), Inches(0.5),
             size=Pt(18), bold=True, color=C_DARK)

    fields = [
        ("time",       "np.ndarray", "时间戳数组，单位秒，shape=(N,)"),
        ("frequency",  "np.ndarray", "基频数组，单位 Hz；静音帧为 NaN"),
        ("confidence", "np.ndarray", "CREPE 置信度，范围 [0, 1]"),
        ("bpm",        "float",      "每分钟节拍数（librosa 估计）"),
    ]
    for i, (name, typ, desc) in enumerate(fields):
        fy = Inches(1.95) + i * Inches(0.88)
        add_rect(slide, rx, fy, Inches(6.5), Inches(0.82),
                 fill=C_WHITE, line=C_LIGHT, line_w=Pt(0.8))
        add_rect(slide, rx, fy, Inches(0.08), Inches(0.82), fill=C_MID)
        add_text(slide, f"{name}  ({typ})",
                 rx + Inches(0.18), fy + Inches(0.05), Inches(6.2), Inches(0.35),
                 size=Pt(13), bold=True, color=C_DARK)
        add_text(slide, desc,
                 rx + Inches(0.18), fy + Inches(0.42), Inches(6.2), Inches(0.35),
                 size=Pt(11), color=C_GRAY)

    add_rect(slide, rx, Inches(5.6), Inches(6.5), Inches(1.0),
             fill=RGBColor(0xE8, 0xF0, 0xFB), line=C_MID, line_w=Pt(1.2))
    add_text(slide, "CREPE 替换 pyin 的收益",
             rx + Inches(0.15), Inches(5.65), Inches(6.2), Inches(0.4),
             size=Pt(14), bold=True, color=C_DARK)
    add_text(slide, "音高提取准确率：~62%（pyin）→ ~80%（CREPE）  ≈ +18%",
             rx + Inches(0.15), Inches(6.05), Inches(6.2), Inches(0.45),
             size=Pt(13), color=C_ACCENT)


# ════════════════════════════════════════════════════════
# Slide 5 — 容错量化设计（徐亦轲 + 钟翔）
# ════════════════════════════════════════════════════════
def slide_quantizer_design(prs):
    slide = blank_slide(prs)
    slide_bg(slide)
    header_bar(slide, "方案设计：BiLSTM-CRF 序列标注 + 边界防御", "Quantizer Design")
    member_tag(slide, "徐亦轲", Inches(8.5), Inches(0.15), C_MID)
    member_tag(slide, "钟翔",   Inches(10.2), Inches(0.15), C_GREEN)

    # ── 左：BiLSTM-CRF 模型结构（徐亦轲）──────────
    lx = Inches(0.4)
    layers = [
        ("输入特征 (4维)\nMIDI编号 · voiced概率 · 置信度 · BPM", C_LIGHT, C_DARK),
        ("BiLSTM × 2层\nhidden=128，双向，Dropout=0.3",          C_MID,   C_WHITE),
        ("全连接层 → 发射矩阵",                                   C_MID,   C_WHITE),
        ("CRF 解码层",                                            C_ACCENT, C_WHITE),
        ("BIO 标签序列  →  音符事件",                             C_GREEN,  C_WHITE),
    ]
    bw, bh, bgap = Inches(5.6), Inches(0.75), Inches(0.15)
    ly_start = Inches(1.35)
    for i, (label, bg, fg) in enumerate(layers):
        by = ly_start + i * (bh + bgap)
        add_rect(slide, lx, by, bw, bh, fill=bg, line=C_MID, line_w=Pt(0.8))
        add_text(slide, label, lx, by + Inches(0.08), bw, bh,
                 size=Pt(13), bold=True, color=fg, align=PP_ALIGN.CENTER)
        if i < len(layers) - 1:
            add_text(slide, "↓",
                     lx + bw / 2 - Inches(0.15), by + bh,
                     Inches(0.3), bgap + Inches(0.04),
                     size=Pt(13), color=C_MID, align=PP_ALIGN.CENTER)

    add_text(slide, "533K 参数  |  完全接口隔离（interfaces.py）",
             lx, Inches(5.65), bw, Inches(0.38),
             size=Pt(12), color=C_GRAY, italic=True)

    # ── 右上：关键工程技巧（徐亦轲）──────────────
    rx = Inches(6.3)
    tricks = [
        ("① 类别不平衡",
         "B类仅占1.2%  →  加权CE辅助损失（B×50）\n+ CRF loss权重0.2，防止模型退化"),
        ("② 八度偏移修正",
         "中位音高比较，自动吸附最近12半音整数倍\n覆盖演唱者高/低八度习惯"),
        ("③ 时间对齐",
         "互相关估算GT与实际发声全局偏移（±1s内）\nper-sample修正BIO标签"),
    ]
    for i, (t, d) in enumerate(tricks):
        ty = Inches(1.35) + i * Inches(1.28)
        add_rect(slide, rx, ty, Inches(6.8), Inches(1.15),
                 fill=C_WHITE, line=C_LIGHT, line_w=Pt(1.0))
        add_rect(slide, rx, ty, Inches(0.08), Inches(1.15), fill=C_ACCENT)
        add_text(slide, t,
                 rx + Inches(0.18), ty + Inches(0.05), Inches(6.5), Inches(0.38),
                 size=Pt(13), bold=True, color=C_DARK)
        add_text(slide, d,
                 rx + Inches(0.18), ty + Inches(0.42), Inches(6.5), Inches(0.7),
                 size=Pt(11), color=C_GRAY)

    # ── 右下：边界防御量化器（钟翔）────────────────
    add_rect(slide, rx, Inches(5.3), Inches(6.8), Inches(1.85),
             fill=RGBColor(0xEF, 0xF7, 0xEF), line=C_GREEN, line_w=Pt(1.5))
    add_rect(slide, rx, Inches(5.3), Inches(0.08), Inches(1.85), fill=C_GREEN)
    add_text(slide, "边界防御量化器（钟翔）",
             rx + Inches(0.18), Inches(5.35), Inches(6.5), Inches(0.42),
             size=Pt(14), bold=True, color=C_DARK)
    add_text(slide,
             "• quantize 接口封装：模型推理 → 失败自动回退 Baseline\n"
             "• 11 项极限单元测试：绝对静音(0Hz) / NaN脏数据 / 空序列等\n"
             "• 全部 100% 通过  →  工业级稳定性保证",
             rx + Inches(0.18), Inches(5.8), Inches(6.5), Inches(1.25),
             size=Pt(12), color=C_GRAY)


# ════════════════════════════════════════════════════════
# Slide 6 — 量化实验结果（徐亦轲）
# ════════════════════════════════════════════════════════
def make_results_chart() -> bytes:
    methods = ["Baseline\n(pyin)", "v1\n(pyin,2k)", "v4\n(CREPE,13k)", "v5\n(CREPE+对齐)"]
    note_acc = [0.308, 0.237, 0.342, 0.510]
    precision = [0.154, 0.233, 0.359, 0.417]
    f1        = [0.199, 0.234, 0.350, 0.453]

    x = np.arange(len(methods))
    w = 0.26

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#F5F8FC")
    ax.set_facecolor("#F5F8FC")

    bars1 = ax.bar(x - w, note_acc, w, label="Note Accuracy", color="#2E75B6", alpha=0.9)
    bars2 = ax.bar(x,     precision, w, label="Precision",     color="#70AD47", alpha=0.9)
    bars3 = ax.bar(x + w, f1,        w, label="F1",            color="#ED7D31", alpha=0.9)

    for bar in [bars1[3], bars2[3], bars3[3]]:
        bar.set_edgecolor("#C00000")
        bar.set_linewidth(2.0)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axhline(0.308 + 0.10, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(3.45, 0.308 + 0.10 + 0.008, "目标线 (baseline+10%)",
            ha="right", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def slide_results(prs):
    slide = blank_slide(prs)
    slide_bg(slide)
    header_bar(slide, "实验结果（HumTrans TEST 集，769 条）", "Experimental Results")
    member_tag(slide, "徐亦轲", Inches(9.0), Inches(0.15), C_MID)

    chart_io = io.BytesIO(make_results_chart())
    slide.shapes.add_picture(chart_io, Inches(0.3), Inches(1.15), Inches(8.2), Inches(4.6))

    rx = Inches(8.7)
    add_text(slide, "关键结论",
             rx, Inches(1.3), Inches(4.3), Inches(0.5),
             size=Pt(18), bold=True, color=C_DARK)

    conclusions = [
        (C_GREEN,  "v5 Note Accuracy\n0.510  (+20.1% vs Baseline)"),
        (C_GREEN,  "F1 Score\n0.453  (+127% vs Baseline)"),
        (C_MID,    "CREPE 替换 pyin\n音高准确率 +18%"),
        (C_MID,    "时间对齐 + 八度修正\n大幅改善标注质量"),
        (C_ACCENT, "目标 ≥ +10%  ✓  达标"),
    ]

    for i, (color, text) in enumerate(conclusions):
        cy = Inches(1.95) + i * Inches(1.02)
        add_rect(slide, rx, cy, Inches(4.3), Inches(0.85),
                 fill=C_WHITE, line=color, line_w=Pt(2.0))
        add_rect(slide, rx, cy, Inches(0.09), Inches(0.85), fill=color)
        add_text(slide, text,
                 rx + Inches(0.18), cy + Inches(0.05), Inches(4.0), Inches(0.78),
                 size=Pt(13), bold=(i == 4), color=C_DARK if i < 4 else color)

    add_text(slide, "可视化工具：streamlit run tools/visualizer.py --server.port 8501",
             Inches(0.4), Inches(6.0), Inches(8.0), Inches(0.38),
             size=Pt(11), color=C_GRAY, italic=True)


# ════════════════════════════════════════════════════════
# Slide 7 — 量化压缩对比（李炎）
# ════════════════════════════════════════════════════════
def make_compression_chart() -> bytes:
    methods = ["低比特标量量化", "矢量量化 (VQ)", "神经网络感知量化"]
    mos      = [2.74, 3.52, 4.31]
    colors   = ["#5B9BD5", "#70AD47", "#ED7D31"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("#F5F8FC")
    ax.set_facecolor("#F5F8FC")

    bars = ax.bar(methods, mos, color=colors, alpha=0.88, width=0.52)
    bars[2].set_edgecolor("#C00000")
    bars[2].set_linewidth(2.2)

    for bar, v in zip(bars, mos):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.04,
                f"{v:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.axhline(4.23, color="#C00000", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.text(2.38, 4.27, "无损基准 4.23", ha="right", fontsize=9, color="#C00000")

    ax.set_ylim(0, 5.0)
    ax.set_ylabel("MOS（5分制）", fontsize=12)
    ax.set_title("三种量化压缩方案 MOS 主观评分对比（n=30）", fontsize=12, pad=8)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.5, -0.15,
            "ANOVA: F(2, 87) = 128.76,  p < 0.001  →  三种方案效果差异极显著",
            ha="center", transform=ax.transAxes, fontsize=10, color="#595959",
            style="italic")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def slide_compression(prs):
    slide = blank_slide(prs)
    slide_bg(slide)
    header_bar(slide, "量化压缩算法对比研究", "Compression Evaluation")
    member_tag(slide, "李炎", Inches(9.0), Inches(0.15), C_PURPLE)

    chart_io = io.BytesIO(make_compression_chart())
    slide.shapes.add_picture(chart_io, Inches(0.3), Inches(1.15), Inches(7.8), Inches(4.8))

    rx = Inches(8.3)
    add_text(slide, "实验设计",
             rx, Inches(1.3), Inches(4.8), Inches(0.48),
             size=Pt(18), bold=True, color=C_DARK)

    design_items = [
        ("被试样本",   "30条独立测试样本（覆盖音高区间/旋律难度）"),
        ("评分方式",   "5分制 MOS 双盲盲听，3×30 评分矩阵"),
        ("评分工具",   "Python scipy 单因素方差分析（ANOVA）"),
        ("统计结果",   "F(2,87) = 128.76，p < 0.001"),
        ("结论",       "感知量化接近无损质量，三方案差异极显著"),
    ]

    for i, (label, desc) in enumerate(design_items):
        dy = Inches(1.95) + i * Inches(1.0)
        add_rect(slide, rx, dy, Inches(4.8), Inches(0.85),
                 fill=C_WHITE, line=C_PURPLE, line_w=Pt(1.8))
        add_rect(slide, rx, dy, Inches(0.09), Inches(0.85), fill=C_PURPLE)
        add_text(slide, label,
                 rx + Inches(0.18), dy + Inches(0.05), Inches(4.5), Inches(0.35),
                 size=Pt(13), bold=True, color=C_DARK)
        add_text(slide, desc,
                 rx + Inches(0.18), dy + Inches(0.42), Inches(4.5), Inches(0.38),
                 size=Pt(11), color=C_GRAY)

    add_rect(slide, Inches(0.3), Inches(6.2), Inches(12.7), Inches(0.95), fill=C_DARK)
    add_text(slide, "  交付物：三种算法完整代码 · 3×30 MOS矩阵 · 统计报告 · 3份可视化图表 · 90条量化音频",
             Inches(0.4), Inches(6.28), Inches(12.5), Inches(0.75),
             size=Pt(14), bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════
# Slide 8 — 风格推断与渲染（窦畅 + 钟翔）
# ════════════════════════════════════════════════════════
def slide_style_render(prs):
    slide = blank_slide(prs)
    slide_bg(slide)
    header_bar(slide, "风格推断与音频渲染", "Style Inference & Rendering")
    member_tag(slide, "窦畅", Inches(8.5), Inches(0.15), C_ACCENT)
    member_tag(slide, "钟翔", Inches(10.2), Inches(0.15), C_GREEN)

    # ── 左：MusicBrain 推断（窦畅）────────────────
    lx = Inches(0.4)
    add_rect(slide, lx, Inches(1.25), Inches(6.0), Inches(0.48), fill=C_ACCENT)
    add_text(slide, "  MusicBrain 推断引擎（窦畅）",
             lx, Inches(1.25), Inches(6.0), Inches(0.48),
             size=Pt(16), bold=True, color=C_WHITE)

    music_cards = [
        ("算法集成",
         ["librosa PiPTrack 频率追踪（FFT 频率估计）",
          "music21 符号引擎（和弦识别）",
          "支持 4 种调性，覆盖 14 种标准三和弦"]),
        ("性能优化",
         ["原始 44.1 kHz → 降至 22.05 kHz",
          "单条推断耗时：1.0s → 0.21s（优化后）",
          "降低 Web 端等待延迟约 79%"]),
        ("风格音色映射",
         ["Pop:  Piano + Electric Piano + Bass",
          "Jazz: Sax + Electric Piano + Acoustic Bass",
          "Classical: Violin + Strings + Cello",
          "Folk: Guitar + Nylon Guitar + Accordion"]),
    ]
    for i, (t, items) in enumerate(music_cards):
        cy = Inches(1.85) + i * Inches(1.7)
        card(slide, lx, cy, Inches(6.0), Inches(1.55), t, items, C_ACCENT)

    # ── 右：FluidSynth 渲染（窦畅 + 钟翔）──────────
    rx = Inches(7.0)
    add_rect(slide, rx, Inches(1.25), Inches(5.9), Inches(0.48), fill=C_MID)
    add_text(slide, "  FluidSynth 底层渲染（窦畅 + 钟翔）",
             rx, Inches(1.25), Inches(5.9), Inches(0.48),
             size=Pt(16), bold=True, color=C_WHITE)

    render_cards = [
        ("离线推拉流模式（窦畅）",
         ["禁用实时音频驱动，切换离线模式",
          "30s 音频合成耗时：0.059s",
          "DLL依赖Hell：libfluidsynth.dll 物理注入 Python 根目录",
          "Git LFS音色库 PCLite.sf2（30MB）完整拉取"]),
        ("Native Subprocess 合成（钟翔）",
         ["重编 FluidSynth 2.5 底层指令（subprocess）",
          "16bit / 44100Hz 高保真渲染",
          "30s 音频合成耗时：0.2688s",
          "渲染性能目标 ≤ 5s，超标 18×"]),
    ]
    for i, (t, items) in enumerate(render_cards):
        cy = Inches(1.85) + i * Inches(2.6)
        col = C_ACCENT if i == 0 else C_GREEN
        card(slide, rx, cy, Inches(5.9), Inches(2.4), t, items, col)

    add_rect(slide, Inches(0.4), Inches(6.58), Inches(12.5), Inches(0.58), fill=C_DARK)
    add_text(slide,
             "  渲染时序：MIDI → FluidSynth 离线合成 → WAV（44100Hz / 16bit） → Gradio 播放",
             Inches(0.5), Inches(6.62), Inches(12.2), Inches(0.5),
             size=Pt(14), bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════
# Slide 9 — 系统界面（窦畅）
# ════════════════════════════════════════════════════════
def slide_ui(prs):
    slide = blank_slide(prs)
    slide_bg(slide)
    header_bar(slide, "系统交互界面", "System UI & Engineering")
    member_tag(slide, "窦畅", Inches(9.0), Inches(0.15), C_ACCENT)

    features = [
        ("黑金风格仪表盘",
         "深色主题 UI，Fetch API 异步 POST 请求\n音频上传与结果返回「零刷新」体验"),
        ("Watchdog 冲突治理",
         "Flask debug 模式监控根目录，temp_upload.wav 触发误重启\n解决方案：use_reloader=False，保证推理任务原子性"),
        ("DLL 依赖地狱解决",
         "FluidSynth Windows 动态链接库路径断裂\n手动追溯符号链接，物理注入 Python 运行根目录"),
        ("Git LFS 资源修复",
         "PCLite.sf2 克隆后仅为 1KB 指针文件\n重新拉取 30MB 二进制 LFS，恢复音色采样完整性"),
    ]

    lx, rx = Inches(0.4), Inches(7.0)
    for i, (title, desc) in enumerate(features):
        x = lx if i % 2 == 0 else rx
        y = Inches(1.4) + (i // 2) * Inches(2.4)
        add_rect(slide, x, y, Inches(6.2), Inches(2.2),
                 fill=C_WHITE, line=C_ACCENT, line_w=Pt(1.5))
        add_rect(slide, x, y, Inches(0.1), Inches(2.2), fill=C_ACCENT)
        add_text(slide, title,
                 x + Inches(0.2), y + Inches(0.1), Inches(5.9), Inches(0.48),
                 size=Pt(16), bold=True, color=C_DARK)
        add_text(slide, desc,
                 x + Inches(0.2), y + Inches(0.62), Inches(5.9), Inches(1.45),
                 size=Pt(13), color=C_GRAY)

    add_rect(slide, Inches(0.4), Inches(6.55), Inches(12.5), Inches(0.62), fill=C_DARK)
    add_text(slide, "  【此处粘贴界面截图】",
             Inches(0.5), Inches(6.58), Inches(12.2), Inches(0.55),
             size=Pt(16), color=C_LIGHT, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════
# Slide 10 — 主观质量评价（窦畅 + 李炎）
# ════════════════════════════════════════════════════════
def make_mos_chart() -> bytes:
    dims = ["音乐合理性", "旋律流畅度", "音准自然度"]
    scores = [4.17, 4.28, 4.24]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    fig.patch.set_facecolor("#F5F8FC")
    ax.set_facecolor("#F5F8FC")

    colors = ["#2E75B6", "#70AD47", "#ED7D31"]
    bars = ax.barh(dims, scores, color=colors, alpha=0.88, height=0.45)
    for bar, v in zip(bars, scores):
        ax.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=12, fontweight="bold")

    ax.axvline(4.23, color="#C00000", linestyle="--", linewidth=1.4)
    ax.text(4.24, 2.5, "均值\n4.23", fontsize=8.5, color="#C00000", va="center")
    ax.set_xlim(3.5, 5.0)
    ax.set_xlabel("评分（5分制）", fontsize=11)
    ax.set_title("主观评分各维度分布（n=30）", fontsize=11, pad=6)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.5, -0.18,
            "σ = 0.68，评分集中稳定",
            ha="center", transform=ax.transAxes, fontsize=9, color="#595959", style="italic")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def slide_mos(prs):
    slide = blank_slide(prs)
    slide_bg(slide)
    header_bar(slide, "主观质量评价（MOS）", "Subjective Evaluation")
    member_tag(slide, "窦畅", Inches(8.5), Inches(0.15), C_ACCENT)
    member_tag(slide, "李炎", Inches(10.2), Inches(0.15), C_PURPLE)

    # ── 左上：MOS 问卷设计（窦畅）──────────────────
    lx = Inches(0.4)
    add_rect(slide, lx, Inches(1.25), Inches(6.0), Inches(0.45), fill=C_ACCENT)
    add_text(slide, "  MOS 问卷设计（窦畅）",
             lx, Inches(1.25), Inches(6.0), Inches(0.45),
             size=Pt(15), bold=True, color=C_WHITE)

    add_rect(slide, lx, Inches(1.75), Inches(6.0), Inches(2.45),
             fill=C_WHITE, line=C_LIGHT, line_w=Pt(0.8))
    add_rect(slide, lx, Inches(1.75), Inches(0.08), Inches(2.45), fill=C_ACCENT)
    dims_dou = [
        "参考标准：ITU-T P.800",
        "① 音色自然度  ② 音高准确度  ③ 节奏稳定性",
        "④ 风格拟合度  ⑤ 整体满意度",
        "3人内部测试：音高准确度评价最高",
        "→ 验证 BiLSTM-CRF 及和弦推断有效性",
    ]
    txBox = slide.shapes.add_textbox(
        lx + Inches(0.18), Inches(1.82), Inches(5.7), Inches(2.3))
    tf = txBox.text_frame
    tf.word_wrap = True
    for j, line in enumerate(dims_dou):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.space_before = Pt(4)
        run = p.add_run()
        run.text = line
        run.font.name = FONT
        run.font.size = Pt(12)
        run.font.color.rgb = C_GRAY

    # ── 左下：图表（李炎）──────────────────────────
    chart_io = io.BytesIO(make_mos_chart())
    slide.shapes.add_picture(chart_io, lx, Inches(4.35), Inches(6.0), Inches(2.85))

    # ── 右：关键指标（李炎）──────────────────────────
    rx = Inches(7.0)
    add_rect(slide, rx, Inches(1.25), Inches(5.9), Inches(0.45), fill=C_PURPLE)
    add_text(slide, "  主观评分结果（李炎）",
             rx, Inches(1.25), Inches(5.9), Inches(0.45),
             size=Pt(15), bold=True, color=C_WHITE)

    metrics = [
        ("4.23 / 5",    "30条样本综合均分",       C_GREEN),
        ("0.46",        "方差（评分高度集中）",    C_MID),
        ("0.68",        "标准差",                 C_MID),
        ("100%",        "30条全部完成评分",        C_GREEN),
        ("音高准确度",   "得分最高维度（BiLSTM验证）", C_ACCENT),
    ]

    for i, (val, label, color) in enumerate(metrics):
        my = Inches(1.85) + i * Inches(1.08)
        add_rect(slide, rx, my, Inches(5.9), Inches(0.95),
                 fill=C_WHITE, line=color, line_w=Pt(2.0))
        add_rect(slide, rx, my, Inches(0.09), Inches(0.95), fill=color)
        add_text(slide, val,
                 rx + Inches(0.2), my + Inches(0.05), Inches(2.5), Inches(0.42),
                 size=Pt(22), bold=True, color=color)
        add_text(slide, label,
                 rx + Inches(0.2), my + Inches(0.5), Inches(5.5), Inches(0.38),
                 size=Pt(12), color=C_GRAY)


# ════════════════════════════════════════════════════════
# Slide 11 — 小结与展望（全组）
# ════════════════════════════════════════════════════════
def slide_summary(prs):
    slide = blank_slide(prs)
    slide_bg(slide)
    header_bar(slide, "小结与展望", "Summary & Future Work")

    # ── 左：已完成 ──────────────────────────────────
    add_rect(slide, Inches(0.4), Inches(1.3), Inches(5.9), Inches(0.5), fill=C_MID)
    add_text(slide, "  ✅  已完成（全组）",
             Inches(0.4), Inches(1.3), Inches(5.9), Inches(0.5),
             size=Pt(17), bold=True, color=C_WHITE)

    done = [
        ("李炎",   "BIO标注 2,000条，Kappa=0.9618（优秀）"),
        ("钟翔",   "DataLoader 6,414条 + 11项边界测试 100%通过"),
        ("徐亦轲", "CREPE F0管线 + BiLSTM-CRF v5，Acc=0.510 +20.1%"),
        ("钟翔",   "FluidSynth 30s合成 0.2688s（目标超标18×）"),
        ("窦畅",   "music21推断 4调性14和弦，耗时0.21s"),
        ("窦畅",   "Flask黑金UI + Fetch API + Watchdog/DLL修复"),
        ("李炎",   "3种压缩算法对比，ANOVA F=128.76，p<0.001"),
        ("窦畅+李炎", "MOS评价体系，综合均分 4.23/5"),
    ]

    member_colors = {
        "徐亦轲": C_MID, "窦畅": C_ACCENT,
        "钟翔": C_GREEN, "李炎": C_PURPLE, "窦畅+李炎": C_PURPLE,
    }

    for i, (member, item) in enumerate(done):
        dy = Inches(1.95) + i * Inches(0.62)
        color = member_colors.get(member, C_MID)
        add_rect(slide, Inches(0.4), dy, Inches(1.1), Inches(0.5), fill=color)
        add_text(slide, member,
                 Inches(0.4), dy, Inches(1.1), Inches(0.5),
                 size=Pt(10), bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
        add_text(slide, f"  {item}",
                 Inches(1.52), dy + Inches(0.06), Inches(4.7), Inches(0.45),
                 size=Pt(12), color=C_DARK)

    # ── 右：下一步 ──────────────────────────────────
    add_rect(slide, Inches(7.0), Inches(1.3), Inches(5.9), Inches(0.5), fill=C_ACCENT)
    add_text(slide, "  🔜  下一步计划",
             Inches(7.0), Inches(1.3), Inches(5.9), Inches(0.5),
             size=Pt(17), bold=True, color=C_WHITE)

    next_steps = [
        "VQ-VAE 风格迁移模块联调（全流程端到端打通）",
        "录制 Demo 视频：哼一段 → 4 种风格输出对比",
        "推理速度优化，达到实时响应 < 2s",
        "MOS 正式测试，扩大到 ≥ 20 人评测样本",
        "（可选）探索 Transformer 替换 LSTM 主干",
    ]
    for i, item in enumerate(next_steps):
        ny = Inches(1.95) + i * Inches(0.78)
        add_rect(slide, Inches(7.0), ny, Inches(5.9), Inches(0.68),
                 fill=C_WHITE, line=C_LIGHT, line_w=Pt(0.8))
        add_rect(slide, Inches(7.0), ny, Inches(0.08), Inches(0.68), fill=C_ACCENT)
        add_text(slide, f"  {item}",
                 Inches(7.1), ny + Inches(0.1), Inches(5.7), Inches(0.5),
                 size=Pt(13), color=C_DARK)

    add_rect(slide, Inches(0.4), Inches(6.48), Inches(12.5), Inches(0.72), fill=C_DARK)
    add_text(slide,
             "各模块已超额完成中期目标，全系统可端到端运行  →  为期末答辩打下坚实基础",
             Inches(0.4), Inches(6.5), Inches(12.5), Inches(0.72),
             size=Pt(16), bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════
def main():
    prs = new_prs()
    slide_cover(prs)             # 1. 封面
    slide_arch(prs)              # 2. 系统架构
    slide_data_engineering(prs)  # 3. 数据工程（钟翔+李炎）
    slide_audio_processing(prs)  # 4. 音频处理（徐亦轲）
    slide_quantizer_design(prs)  # 5. 容错量化设计（徐亦轲+钟翔）
    slide_results(prs)           # 6. 量化实验结果（徐亦轲）
    slide_compression(prs)       # 7. 量化压缩对比（李炎）
    slide_style_render(prs)      # 8. 风格推断与渲染（窦畅+钟翔）
    slide_ui(prs)                # 9. 系统界面（窦畅）
    slide_mos(prs)               # 10. 主观质量评价（窦畅+李炎）
    slide_summary(prs)           # 11. 小结与展望

    out = Path("reports/midterm_group.pptx")
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"已生成：{out.resolve()}")
    print(f"共 11 页：封面 / 架构 / 数据工程 / 音频处理 / 量化设计 / 量化结果 / 压缩对比 / 风格渲染 / 界面 / MOS / 小结")


if __name__ == "__main__":
    main()
