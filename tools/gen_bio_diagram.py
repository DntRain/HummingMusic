"""
tools/gen_bio_diagram.py - 生成 BIO 标注方案示意图
输出：reports/bio_annotation_diagram.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from pathlib import Path

# ── 字体 ──────────────────────────────────────
for fp in ["/usr/share/fonts/WindowsFonts/msyh.ttc",
           "/usr/share/fonts/WindowsFonts/msyhbd.ttc"]:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

# ── 参数 ──────────────────────────────────────
STEP_S = 0.01          # 10ms / frame
N = 280                # 280 frames = 2.8s

# ── 四个音符（MIDI pitch, start_frame, end_frame）
NOTES = [
    (60, 20,  58),   # C4 ~261 Hz
    (62, 78, 118),   # D4 ~293 Hz
    (64, 145, 172),  # E4 ~329 Hz
    (60, 192, 240),  # C4 ~261 Hz
]
NOTE_NAMES = {60: "C4", 62: "D4", 64: "E4"}
NOTE_COLORS = ["#E74C3C", "#F39C12", "#2ECC71", "#E74C3C"]

def pitch_to_hz(p):
    return 440.0 * 2 ** ((p - 69) / 12)

# ── BIO 标签 ──────────────────────────────────
labels = np.zeros(N, dtype=int)
for pitch, s, e in NOTES:
    labels[s] = 1
    if e > s + 1:
        labels[s+1:e] = 2

# ── 模拟 F0 曲线（带抖动+静音段=NaN）
rng = np.random.default_rng(42)
freq = np.full(N, np.nan)
for pitch, s, e in NOTES:
    hz = pitch_to_hz(pitch)
    jitter = rng.normal(0, hz * 0.008, e - s)
    freq[s:e] = hz + jitter

time = np.arange(N) * STEP_S

# ── 模拟置信度 ────────────────────────────────
conf = rng.uniform(0.05, 0.25, N)
for _, s, e in NOTES:
    conf[s:e] = rng.uniform(0.82, 0.99, e - s)

# ── 颜色 ──────────────────────────────────────
CLR_O = "#BDC3C7"
CLR_B = "#E74C3C"
CLR_I = "#3498DB"

# ══════════════════════════════════════════════
# 图形布局
# ══════════════════════════════════════════════
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("#F8F9FA")

gs = fig.add_gridspec(
    4, 1,
    height_ratios=[2.2, 1.5, 0.7, 1.6],
    hspace=0.45,
    left=0.10, right=0.93, top=0.90, bottom=0.06,
)
ax_piano = fig.add_subplot(gs[0])
ax_f0    = fig.add_subplot(gs[1], sharex=ax_piano)
ax_bio   = fig.add_subplot(gs[2], sharex=ax_piano)
ax_conf  = fig.add_subplot(gs[3], sharex=ax_piano)

fig.text(0.5, 0.96, "BIO 标注方案示意图",
         ha="center", fontsize=20, fontweight="bold", color="#1F3864")
fig.text(0.5, 0.93,
         "B（音符起始）· I（音符持续）· O（静音）  |  帧步长 10ms  |  输入特征 4 维",
         ha="center", fontsize=12, color="#595959")

# ─── Panel 1: Piano Roll ──────────────────────
ax_piano.set_facecolor("#1A2535")
pitches_present = sorted({p for p, *_ in NOTES})
pitch_range = range(min(pitches_present)-3, max(pitches_present)+4)

for y in pitch_range:
    c = "#232F3E" if y % 2 == 0 else "#1A2535"
    ax_piano.axhspan(y - 0.5, y + 0.5, color=c, zorder=0)

for i, (pitch, s, e) in enumerate(NOTES):
    ts, te = s * STEP_S, e * STEP_S
    rect = mpatches.FancyBboxPatch(
        (ts, pitch - 0.45), te - ts, 0.9,
        boxstyle="round,pad=0.01",
        facecolor=NOTE_COLORS[i], edgecolor="white",
        linewidth=1.2, alpha=0.92, zorder=3,
    )
    ax_piano.add_patch(rect)
    ax_piano.text((ts+te)/2, pitch, NOTE_NAMES.get(pitch, str(pitch)),
                  ha="center", va="center", fontsize=10,
                  fontweight="bold", color="white", zorder=4)

ax_piano.set_ylim(min(pitch_range)-0.5, max(pitch_range)+0.5)
ax_piano.set_yticks(pitches_present)
ax_piano.set_yticklabels([NOTE_NAMES.get(p, str(p)) for p in pitches_present],
                          fontsize=11, color="white")
ax_piano.tick_params(colors="white", labelcolor="white")
for sp in ax_piano.spines.values():
    sp.set_color("#334455")
ax_piano.set_ylabel("音高\n(MIDI)", fontsize=11, color="white")
ax_piano.set_title("① 钢琴卷帘（Ground Truth MIDI）", fontsize=13,
                    fontweight="bold", color="white", pad=6)

# ─── Panel 2: F0 曲线 ────────────────────────
ax_f0.set_facecolor("#FAFBFC")
silence_mask = np.isnan(freq)
in_sil, s_start = False, 0
for i in range(N + 1):
    is_s = (i == N) or silence_mask[i]
    if is_s and not in_sil:
        in_sil, s_start = True, i
    elif not is_s and in_sil:
        in_sil = False
        ax_f0.axvspan(s_start*STEP_S, i*STEP_S, color="#E8ECF0", alpha=0.9, zorder=1)

for pitch in set(p for p, *_ in NOTES):
    ax_f0.axhline(pitch_to_hz(pitch), color="#CDD3DA", linewidth=1,
                   linestyle="--", zorder=1)

for i, (pitch, s, e) in enumerate(NOTES):
    ts = np.arange(s, e) * STEP_S
    ax_f0.plot(ts, freq[s:e], color=NOTE_COLORS[i], linewidth=2.0, zorder=3)

ax_f0.text(0.63, 0.50, "静音段 (freq = NaN)",
           transform=ax_f0.transAxes, fontsize=10, color="#95A5A6",
           ha="center", va="center",
           bbox=dict(boxstyle="round,pad=0.3", fc="#E8ECF0", ec="#CBD3DA", alpha=0.9))

ax_f0.set_ylim(220, 380)
ax_f0.set_yticks([261, 293, 330])
ax_f0.set_yticklabels(["C4\n261Hz", "D4\n293Hz", "E4\n330Hz"], fontsize=9)
ax_f0.set_ylabel("频率 (Hz)", fontsize=11)
ax_f0.set_title("② CREPE F0 提取（含音高抖动）", fontsize=13, fontweight="bold", pad=5)
ax_f0.grid(axis="y", alpha=0.3, linestyle=":")

# ─── Panel 3: BIO 标签 ───────────────────────
ax_bio.set_facecolor("#FAFBFC")
COLORS = {0: CLR_O, 1: CLR_B, 2: CLR_I}
for i in range(N):
    ax_bio.axvspan(i*STEP_S, (i+1)*STEP_S, color=COLORS[labels[i]], alpha=0.88)
for _, s, e in NOTES:
    ax_bio.axvline(s*STEP_S, color="#C0392B", linewidth=2.0, zorder=3)

legend_patches = [
    mpatches.Patch(facecolor=CLR_B, label="B（起始帧）— 音符开始的第一帧"),
    mpatches.Patch(facecolor=CLR_I, label="I（持续帧）— 音符仍在延续"),
    mpatches.Patch(facecolor=CLR_O, label="O（静音帧）— 音符间隔或静音"),
]
ax_bio.set_yticks([])
ax_bio.set_ylabel("BIO\n标签", fontsize=11)
ax_bio.set_title("③ 帧级 BIO 标注序列（红色竖线 = B 帧位置）", fontsize=13,
                  fontweight="bold", pad=5)
ax_bio.legend(handles=legend_patches, loc="upper right",
               fontsize=10, ncol=3, framealpha=0.9, edgecolor="#CBD3DA")

# ─── Panel 4: 置信度 ─────────────────────────
ax_conf.set_facecolor("#FAFBFC")
for _, s, e in NOTES:
    ax_conf.axvspan(s*STEP_S, e*STEP_S, color="#EBF5FB", alpha=0.6, zorder=0)
ax_conf.fill_between(time, conf, alpha=0.35, color="#2E75B6", zorder=1)
ax_conf.plot(time, conf, color="#2E75B6", linewidth=1.2, zorder=2)
ax_conf.axhline(0.80, color="#E74C3C", linewidth=1.5,
                 linestyle="--", zorder=3, label="置信度阈值 0.80")
ax_conf.set_ylim(0, 1.15)
ax_conf.set_ylabel("置信度", fontsize=11)
ax_conf.set_xlabel("时间 (秒)", fontsize=12)
ax_conf.set_title("④ CREPE 置信度（< 0.80 → freq = NaN，帧标为 O）", fontsize=13,
                   fontweight="bold", pad=5)
ax_conf.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax_conf.grid(axis="y", alpha=0.3, linestyle=":")

# ── 时间轴格式 ────────────────────────────────
for ax in [ax_piano, ax_f0, ax_bio, ax_conf]:
    ax.set_xlim(0, N * STEP_S)
    ax.tick_params(axis="x", labelsize=10)
plt.setp(ax_piano.get_xticklabels(), visible=False)
plt.setp(ax_f0.get_xticklabels(), visible=False)
plt.setp(ax_bio.get_xticklabels(), visible=False)

# ── 右侧：类别分布说明 ────────────────────────
n_B = int((labels == 1).sum())
n_I = int((labels == 2).sum())
n_O = int((labels == 0).sum())
note_text = (
    "类别分布（本示例）\n"
    f"B: {n_B}/{N} = {n_B/N*100:.1f}%\n"
    f"I: {n_I}/{N} = {n_I/N*100:.1f}%\n"
    f"O: {n_O}/{N} = {n_O/N*100:.1f}%\n"
    "\n"
    "真实数据中\n"
    "B 类仅 ~1.2%\n"
    "→ 严重类别不平衡\n"
    "→ 加权 CE 损失\n"
    "   (B × 50)"
)
fig.text(0.945, 0.50, note_text,
         ha="left", va="center", fontsize=10, color="#2C3E50",
         bbox=dict(boxstyle="round,pad=0.6", fc="#FEF9E7",
                   ec="#F39C12", linewidth=1.8))

# ── 左侧：特征说明 ────────────────────────────
feat_text = (
    "模型输入特征\n(4 维 / 帧)\n"
    "─────────────────\n"
    "[0] MIDI 音符编号\n"
    "      静音帧填 0\n"
    "[1] voiced 标志\n"
    "      0 or 1\n"
    "[2] CREPE 置信度\n"
    "      范围 [0, 1]\n"
    "[3] 相对节拍位置\n"
    "      (t mod Tbeat)\n"
    "      / Tbeat"
)
fig.text(0.01, 0.50, feat_text,
         ha="left", va="center", fontsize=10, color="#2C3E50",
         bbox=dict(boxstyle="round,pad=0.6", fc="#EAF4FB",
                   ec="#2E75B6", linewidth=1.8))

# ── 保存 ──────────────────────────────────────
Path("reports").mkdir(exist_ok=True)
out = "reports/bio_annotation_diagram.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"已生成: {out}")
