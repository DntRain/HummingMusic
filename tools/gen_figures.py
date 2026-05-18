"""
tools/gen_figures.py - 汇总实验数据并生成全部交付图表

读取 logs/ 与 reports/ 下原始数据，解析后：
  1. 写出汇总 JSON: reports/week11/data/metrics.json
  2. 生成 8 张交付图表到 reports/week11/figures/

用法：
    PYTHONPATH=. python tools/gen_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
LOGS_DIR   = ROOT / "logs"
FIG_DIR    = ROOT / "reports" / "week11" / "figures"
DATA_DIR   = ROOT / "reports" / "week11" / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ─────────────────────────────────────────────────────────
# 日志解析
# ─────────────────────────────────────────────────────────

_RE_V5 = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+loss=([\d.]+)\s+\|\s+val_acc=([\d.]+)"
)
_RE_VQVAE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\|\s+train\s+recon=([\d.]+)\s+vq=([\d.]+)\s+\|\s+"
    r"val\s+recon=([\d.]+)\s+vq=([\d.]+)"
)
_RE_FINETUNE = re.compile(
    r"\[(\w+)\]\s+Epoch\s+(\d+)/\d+\s+loss=([\d.]+)"
)


def parse_v5(log_path: Path) -> dict:
    epochs, losses, accs = [], [], []
    for line in log_path.read_text().splitlines():
        m = _RE_V5.search(line)
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            accs.append(float(m.group(3)))
    return {"epoch": epochs, "loss": losses, "val_acc": accs}


def parse_vqvae(log_path: Path) -> dict:
    e, tr, tv, vr, vv = [], [], [], [], []
    for line in log_path.read_text().splitlines():
        m = _RE_VQVAE.search(line)
        if m:
            e.append(int(m.group(1)))
            tr.append(float(m.group(2)))
            tv.append(float(m.group(3)))
            vr.append(float(m.group(4)))
            vv.append(float(m.group(5)))
    return {"epoch": e, "train_recon": tr, "train_vq": tv,
            "val_recon": vr, "val_vq": vv}


def parse_finetune(log_path: Path) -> dict[str, dict]:
    by_style: dict[str, dict] = {}
    for line in log_path.read_text().splitlines():
        m = _RE_FINETUNE.search(line)
        if m:
            style, ep, loss = m.group(1), int(m.group(2)), float(m.group(3))
            d = by_style.setdefault(style, {"epoch": [], "loss": []})
            d["epoch"].append(ep)
            d["loss"].append(loss)
    return by_style


# ─────────────────────────────────────────────────────────
# 汇总指标
# ─────────────────────────────────────────────────────────

QUANTIZER_VERSIONS = {
    "Baseline (pyin)":         {"note_acc": 0.308, "precision": 0.154, "f1": 0.199},
    "v1 (pyin, 2k)":           {"note_acc": 0.237, "precision": 0.233, "f1": 0.234},
    "v2 (CREPE, 13k)":         {"note_acc": 0.272, "precision": 0.288, "f1": 0.279},
    "v3 (+align)":             {"note_acc": 0.295, "precision": 0.310, "f1": 0.300},
    "v4 (+50ep)":              {"note_acc": 0.342, "precision": 0.359, "f1": 0.350},
    "v5 (octave-fix)":         {"note_acc": 0.5095, "precision": 0.4165, "f1": 0.4528},
}


def load_ablation() -> dict:
    p = ROOT / "reports" / "week10" / "ablation_results.json"
    return json.loads(p.read_text())


# ─────────────────────────────────────────────────────────
# 图表生成
# ─────────────────────────────────────────────────────────

STYLE_COLORS = {
    "pop":       "#e74c3c",
    "jazz":      "#f39c12",
    "classical": "#3498db",
    "folk":      "#27ae60",
}


def fig1_v5_training(v5: dict, out: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    color1 = "#1f77b4"
    ax1.plot(v5["epoch"], v5["loss"], color=color1, marker="o",
             markersize=3, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (CRF + weighted CE)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.plot(v5["epoch"], v5["val_acc"], color=color2, marker="s",
             markersize=3, label="Validation Note Accuracy")
    ax2.set_ylabel("Note Accuracy", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, max(v5["val_acc"]) * 1.15)
    ax2.grid(False)

    best = max(v5["val_acc"])
    best_ep = v5["epoch"][v5["val_acc"].index(best)]
    ax2.axhline(best, color=color2, ls="--", alpha=0.4)
    ax2.annotate(f"best={best:.4f} @ ep{best_ep}",
                 xy=(best_ep, best), xytext=(best_ep + 1, best - 0.05),
                 fontsize=9, color=color2,
                 arrowprops=dict(arrowstyle="->", color=color2, alpha=0.6))

    plt.title("Fig 1. BiLSTM-CRF v5 Training Curve (HumTrans, 30 epochs)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig2_vqvae_training(vq: dict, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(vq["epoch"], vq["train_recon"], label="Train", color="#1f77b4")
    ax1.plot(vq["epoch"], vq["val_recon"],   label="Validation",
             color="#d62728", ls="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Reconstruction Loss (BCE)")
    ax1.set_title("Reconstruction Loss")
    ax1.set_yscale("log")
    ax1.legend()

    ax2.plot(vq["epoch"], vq["train_vq"], label="Train", color="#2ca02c")
    ax2.plot(vq["epoch"], vq["val_vq"],   label="Validation",
             color="#ff7f0e", ls="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("VQ Loss (codebook + β·commit)")
    ax2.set_title("VQ Loss")
    ax2.legend()

    fig.suptitle("Fig 2. VQ-VAE v1 Training Curves (50 epochs, 70k MIDI segments)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig3_finetune(ft: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for style, data in ft.items():
        ax.plot(data["epoch"], data["loss"], marker="o", markersize=3,
                label=style.capitalize(), color=STYLE_COLORS.get(style, "gray"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss (BCE)")
    ax.set_yscale("log")
    ax.set_title("Fig 3. Style-Specific Decoder Fine-tune (30 epochs × 4 styles)")
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig4_quantizer_compare(versions: dict, out: Path) -> None:
    labels = list(versions.keys())
    acc = [versions[k]["note_acc"]  for k in labels]
    prec = [versions[k]["precision"] for k in labels]
    f1   = [versions[k]["f1"]        for k in labels]

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, acc,  w, label="Note Accuracy", color="#1f77b4")
    ax.bar(x,     prec, w, label="Precision",     color="#2ca02c")
    ax.bar(x + w, f1,   w, label="F1",            color="#d62728")

    for xi, v in zip(x - w, acc):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x,     prec):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + w, f1):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Fig 4. Quantizer Version Comparison (HumTrans TEST, 769 samples)")
    ax.set_ylim(0, 0.62)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig5_ablation_bars(abl: dict, out: Path) -> None:
    cfgs   = list(abl["results"].keys())
    labels = [c.replace("(baseline)", "(BL)") for c in cfgs]
    prec   = [abl["results"][c]["precision"]      for c in cfgs]
    rec    = [abl["results"][c]["recall"]         for c in cfgs]
    f1     = [abl["results"][c]["f1"]             for c in cfgs]
    acc    = [abl["results"][c]["note_accuracy"]  for c in cfgs]

    x = np.arange(len(cfgs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 1.5 * w, prec, w, label="Precision",     color="#2ca02c")
    ax.bar(x - 0.5 * w, rec,  w, label="Recall",        color="#ff7f0e")
    ax.bar(x + 0.5 * w, f1,   w, label="F1",            color="#9467bd")
    ax.bar(x + 1.5 * w, acc,  w, label="Note Accuracy", color="#1f77b4")

    for xi, v in zip(x + 1.5 * w, acc):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Fig 5. Ablation Study on TEST ({abl['n_samples']} samples)")
    ax.set_ylim(0, max(acc) * 1.18)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig6_ablation_delta(abl: dict, out: Path) -> None:
    cfgs   = list(abl["results"].keys())
    base   = abl["results"][cfgs[0]]["note_accuracy"]
    delta  = [abl["results"][c]["note_accuracy"] - base for c in cfgs]
    labels = [c.replace("(baseline)", "(BL)") for c in cfgs]
    colors = ["#1f77b4" if d == 0 else "#d62728" for d in delta]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, delta, color=colors)
    for b, d in zip(bars, delta):
        ha = "left" if d >= 0 else "right"
        ax.text(d, b.get_y() + b.get_height() / 2,
                f"  Δ={d:+.4f}  ", va="center", ha=ha, fontsize=9)

    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Δ Note Accuracy vs Full v5 Baseline")
    ax.set_title("Fig 6. Per-Module Contribution (Ablation Δ)")
    ax.invert_yaxis()
    margin = max(abs(min(delta)), abs(max(delta))) * 0.18
    ax.set_xlim(min(delta) - margin, max(delta) + margin + 0.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig7_dataset_pie(out: Path) -> None:
    sizes  = [13080, 765, 769]
    labels = [f"TRAIN\n{sizes[0]:,}", f"VALID\n{sizes[1]:,}", f"TEST\n{sizes[2]:,}"]
    colors = ["#3498db", "#f39c12", "#27ae60"]
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontweight("bold")
    ax.set_title(f"Fig 7. HumTrans Dataset Split (Total {sum(sizes):,} samples)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig8_pipeline_summary(quant: dict, abl: dict, ft: dict, out: Path) -> None:
    """端到端 pipeline 4 个阶段的关键指标卡片图。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Fig 8. End-to-End Pipeline Summary", fontsize=13, y=1.00)

    # (a) Quantizer evolution
    ax = axes[0, 0]
    labels = list(quant.keys())
    accs   = [quant[k]["note_acc"] for k in labels]
    ax.plot(range(len(labels)), accs, marker="o", color="#1f77b4", lw=2)
    for i, a in enumerate(accs):
        ax.text(i, a + 0.012, f"{a:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([k.split(" ")[0] for k in labels], rotation=20, fontsize=8)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("Note Accuracy")
    ax.set_title("(a) Quantizer iteration  (Baseline → v5: +20.1%)")

    # (b) Ablation deltas
    ax = axes[0, 1]
    cfgs  = list(abl["results"].keys())
    base  = abl["results"][cfgs[0]]["note_accuracy"]
    delta = [abl["results"][c]["note_accuracy"] - base for c in cfgs[1:]]
    names = [c.replace("w/o ", "") for c in cfgs[1:]]
    colors = ["#d62728" if d < -0.1 else "#ff7f0e" if d < -0.005 else "#1f77b4"
              for d in delta]
    ax.barh(names, delta, color=colors)
    xmin = min(delta) * 1.20
    ax.set_xlim(xmin, 0.05)
    for i, d in enumerate(delta):
        ax.text(0.01, i, f"{d:+.4f}", va="center", ha="left", fontsize=9,
                fontweight="bold")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("Δ Note Accuracy")
    ax.set_title("(b) Module ablation impact")
    ax.invert_yaxis()

    # (c) VQ-VAE convergence (final loss per style)
    ax = axes[1, 0]
    styles = list(ft.keys())
    finals = [ft[s]["loss"][-1] for s in styles]
    colors_ = [STYLE_COLORS.get(s, "gray") for s in styles]
    ax.bar(styles, finals, color=colors_)
    for i, v in enumerate(finals):
        ax.text(i, v + max(finals) * 0.02, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.set_ylabel("Final Fine-tune Loss (BCE)")
    ax.set_title("(c) Style-specific decoder final loss")

    # (d) Pipeline metrics summary box
    ax = axes[1, 1]
    ax.axis("off")
    v5     = quant["v5 (octave-fix)"]
    abl_bl = abl["results"]["Full v5 (baseline)"]
    text = (
        "Pipeline End-to-End Summary\n"
        "─────────────────────────────\n"
        f"Quantizer (v5 on TEST):\n"
        f"  Note Acc  = {v5['note_acc']:.4f}\n"
        f"  Precision = {v5['precision']:.4f}\n"
        f"  F1        = {v5['f1']:.4f}\n\n"
        f"Ablation (re-evaluated):\n"
        f"  Baseline NoteAcc = {abl_bl['note_accuracy']:.4f}\n"
        f"  Predicted notes  = {abl_bl['n_pred']:,}\n"
        f"  GT notes         = {abl_bl['n_gt']:,}\n\n"
        f"Style Transfer:\n"
        f"  4 styles × decoder fine-tune\n"
        f"  Final loss range: "
        f"{min(finals):.4f} – {max(finals):.4f}"
    )
    ax.text(0.02, 0.97, text, transform=ax.transAxes,
            fontsize=10, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.6",
                      facecolor="#f5f5f5", edgecolor="#cccccc"))
    ax.set_title("(d) Numerical summary")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ─────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────

def main() -> None:
    # 1. 解析日志
    v5 = parse_v5(LOGS_DIR / "train_v5_crepe_octave.log")
    vq = parse_vqvae(LOGS_DIR / "train_vqvae.log")
    ft = parse_finetune(LOGS_DIR / "finetune_decoder_v2.log")
    abl = load_ablation()

    # 2. 写汇总 metrics.json
    metrics = {
        "quantizer_versions": QUANTIZER_VERSIONS,
        "ablation": abl,
        "training_curves": {
            "bilstm_crf_v5": v5,
            "vqvae_v1":      vq,
            "decoder_finetune": ft,
        },
        "dataset": {"train": 13080, "valid": 765, "test": 769},
        "models": {
            "quantizer_v5_params": 533266,
            "vqvae_v1_params":     420080,
            "codebook_size":       512,
            "embedding_dim":       64,
        },
    }
    out_json = DATA_DIR / "metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[ok] 写汇总指标 → {out_json}")

    # 3. 生成所有图表
    fig1_v5_training(v5,             FIG_DIR / "fig1_bilstm_crf_v5_training.png")
    fig2_vqvae_training(vq,          FIG_DIR / "fig2_vqvae_training.png")
    fig3_finetune(ft,                FIG_DIR / "fig3_decoder_finetune.png")
    fig4_quantizer_compare(QUANTIZER_VERSIONS,
                                     FIG_DIR / "fig4_quantizer_comparison.png")
    fig5_ablation_bars(abl,          FIG_DIR / "fig5_ablation_bars.png")
    fig6_ablation_delta(abl,         FIG_DIR / "fig6_ablation_delta.png")
    fig7_dataset_pie(                FIG_DIR / "fig7_dataset_split.png")
    fig8_pipeline_summary(QUANTIZER_VERSIONS, abl, ft,
                                     FIG_DIR / "fig8_pipeline_summary.png")

    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"[ok] {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.1f} KB)")
    print(f"\n全部 {len(list(FIG_DIR.glob('*.png')))} 张图表生成完成")


if __name__ == "__main__":
    main()
