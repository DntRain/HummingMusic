"""
读取 bench_latency 输出 JSON，生成可视化图与 Markdown 延迟测试报告。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

matplotlib.use("Agg")

for fp in [
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
    "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
]:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

STAGES = ["extract_pitch", "quantize_humming", "transfer_style", "render_audio"]
STAGE_COLORS = ["#C44E52", "#4C72B0", "#55A868", "#CCB974"]
TARGET_SEC = 10.0


def fig_stage_pie(stage_means: dict[str, float], out: Path):
    vals = [stage_means[s] for s in STAGES]
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        vals, labels=STAGES, colors=STAGE_COLORS, autopct="%1.1f%%",
        startangle=90, textprops=dict(fontsize=10))
    total = sum(vals)
    ax.set_title(f"各阶段延迟占比（总均值 {total:.2f}s）")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def fig_stage_bar(stage_summary: dict, out: Path):
    means = [stage_summary[s]["mean"] for s in STAGES]
    stds = [stage_summary[s]["stdev"] for s in STAGES]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(STAGES, means, yerr=stds, capsize=5,
                  color=STAGE_COLORS, alpha=0.85)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2,
                m + max(means) * 0.02,
                f"{m:.3f}s", ha="center", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("延迟 (秒, log)")
    ax.set_title("各阶段延迟均值 ± 1σ（对数纵轴）")
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def fig_case_total(per_case: list, out: Path):
    labels = [f"{Path(c['audio']).name}\n{c['style']}" for c in per_case]
    means = [c["summary"]["total"]["mean"] for c in per_case]
    stds = [c["summary"]["total"]["stdev"] for c in per_case]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=4,
                  color="#4C72B0", alpha=0.85)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.2,
                f"{m:.2f}", ha="center", fontsize=9)
    ax.axhline(TARGET_SEC, color="red", linestyle="--", linewidth=1,
               label=f"目标 ≤ {TARGET_SEC:.0f}s")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("总延迟 (秒)")
    ax.set_title("每个 case 的总延迟（音频 × 风格）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def fig_stage_stack(per_case: list, out: Path):
    labels = [f"{Path(c['audio']).name}\n{c['style']}" for c in per_case]
    bottom = np.zeros(len(per_case))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for s, color in zip(STAGES, STAGE_COLORS):
        vals = np.array([c["summary"][s]["mean"] for c in per_case])
        ax.bar(range(len(labels)), vals, bottom=bottom, color=color,
               label=s, edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.axhline(TARGET_SEC, color="red", linestyle="--", linewidth=1,
               label=f"目标 ≤ {TARGET_SEC:.0f}s")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("延迟 (秒)")
    ax.set_title("每个 case 各阶段延迟堆叠")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def write_report(data: dict, fig_paths: list[Path], out: Path,
                 env_note: str, baseline: dict | None = None):
    meta = data["meta"]
    overall = data["per_stage_overall"]
    total_mean = overall["total"]["mean"]
    target_pass = total_mean <= TARGET_SEC

    lines = []
    lines.append("# Week12 端到端延迟测试报告\n")
    lines.append("## 1. 测试环境与配置\n")
    lines.append(f"- 测试音频：{', '.join(meta['audios'])}")
    lines.append(f"- 测试风格：{', '.join(meta['styles'])}")
    lines.append(f"- 每 case 跑数：{meta['runs']}（warmup {meta['warmup']}）")
    lines.append(f"- 性能目标：端到端 ≤ **{TARGET_SEC:.0f}s**")
    lines.append(env_note + "\n")

    lines.append("## 2. 总体延迟\n")
    lines.append(f"- **当前均值**：{total_mean:.3f} s（中位 "
                 f"{overall['total']['median']:.3f}s，最大 "
                 f"{overall['total']['max']:.3f}s）")
    status = "✅ PASS" if target_pass else "❌ FAIL"
    gap = total_mean - TARGET_SEC
    lines.append(f"- **达标情况**：{status}（差距 "
                 f"{'+' if gap >= 0 else ''}{gap:.2f}s）\n")

    lines.append("## 3. 各阶段延迟分解\n")
    lines.append("| 阶段 | 均值 (s) | 中位 (s) | 标准差 (s) | 最大 (s) | 占比 |")
    lines.append("|------|----------|----------|------------|----------|------|")
    for s in STAGES + ["total"]:
        st = overall[s]
        pct = (st["mean"] / overall["total"]["mean"] * 100
               if s != "total" else 100.0)
        lines.append(f"| {s} | {st['mean']:.3f} | {st['median']:.3f} | "
                     f"{st['stdev']:.3f} | {st['max']:.3f} | {pct:.1f}% |")
    lines.append("")

    lines.append("## 4. 可视化\n")
    titles = [
        "图 1 各阶段延迟占比",
        "图 2 各阶段延迟均值 ± 1σ（对数坐标）",
        "图 3 每个 case 总延迟",
        "图 4 每个 case 各阶段延迟堆叠",
        "图 5 基线 vs 优化对比",
    ]
    for t, p in zip(titles, fig_paths):
        lines.append(f"### {t}\n")
        lines.append(f"![]({p.relative_to(out.parent).as_posix()})\n")

    if baseline is not None:
        bo = baseline["per_stage_overall"]
        lines.append("## 4.5 基线 vs 优化对比\n")
        lines.append("| 阶段 | 基线均值 (s) | 优化均值 (s) | 加速比 | 节省 (s) |")
        lines.append("|------|--------------|--------------|--------|----------|")
        for s in STAGES + ["total"]:
            bm = bo[s]["mean"]
            om = overall[s]["mean"]
            sp = bm / om if om > 0 else float("inf")
            lines.append(f"| {s} | {bm:.3f} | {om:.3f} | "
                         f"{sp:.2f}× | {bm - om:.3f} |")
        lines.append("")

    lines.append("## 5. 瓶颈分析\n")
    ratios = {s: overall[s]["mean"] / overall["total"]["mean"] for s in STAGES}
    bottleneck = max(ratios, key=ratios.get)
    lines.append(f"- **唯一瓶颈**：`{bottleneck}` 占总延迟 "
                 f"**{ratios[bottleneck] * 100:.1f}%**"
                 f"（{overall[bottleneck]['mean']:.2f}s）。")
    lines.append("- 其余三阶段合计占比 "
                 f"**{(1 - ratios[bottleneck]) * 100:.1f}%**，"
                 "几乎不消耗预算，**优化必须聚焦音高提取**。")
    lines.append("- 当前 `extract_pitch` 在本机走 librosa.pyin 路径"
                 "（CREPE/TensorFlow 在本机无法安装，pipeline 已加 fallback），"
                 "生产部署机改走 CREPE 后延迟分布可能不同，需复测。\n")

    lines.append("## 6. 优化路径建议（按预期收益降序）\n")
    lines.append("1. **GPU CREPE / torchcrepe**：本机已有 RTX 4060 + PyTorch，"
                 "改用 `torchcrepe` 可绕过 TensorFlow 依赖，并启用 GPU；"
                 "经验值可将 12s 级 pyin 压到 1-2s。预期单点收益 ≥ 8s。")
    lines.append("2. **音频裁切 + VAD**：在 `extract_pitch` 前裁去前后静音段，"
                 "对 6s+ 音频可减少 20-40% 计算量。")
    lines.append("3. **pyin 参数松弛**（fallback 路径）：增大 hop_length 至 "
                 "20ms / 缩小 frame_length，可换取约 30-50% 加速，"
                 "精度损失需用 quantizer 测试集回测。")
    lines.append("4. **流式 / 分块预热**：录音过程中边录边提取，"
                 "用户感知延迟可降至「最后一块 + 后续阶段」。")
    lines.append("5. **量化模型权重 lazy import**：当前导入接口即触发模型加载，"
                 "首次冷启动延迟未计入；可考虑首屏预加载。\n")

    lines.append("## 7. 下一步\n")
    lines.append("- 安装 `torchcrepe` 并接入 `audio_processing._extract_f0`；")
    lines.append("- 复跑 benchmark 验证；")
    lines.append("- 与钟翔同步部署机的 CREPE 版本测量结果，对齐基线。\n")

    out.write_text("\n".join(lines), encoding="utf-8")


def fig_compare(baseline: dict, optimized: dict, out: Path):
    labels = STAGES + ["total"]
    bm = [baseline["per_stage_overall"][k]["mean"] for k in labels]
    om = [optimized["per_stage_overall"][k]["mean"] for k in labels]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w / 2, bm, w, label="基线 (pyin)", color="#C44E52", alpha=0.85)
    b2 = ax.bar(x + w / 2, om, w, label="优化 (torchcrepe-GPU)",
                color="#55A868", alpha=0.85)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.1,
                    f"{h:.2f}s", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("延迟 (秒, log)")
    ax.set_title("基线 vs 优化：各阶段延迟对比")
    ax.axhline(TARGET_SEC, color="black", linestyle="--",
               linewidth=0.8, label=f"目标 {TARGET_SEC:.0f}s")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--baseline", help="若提供则生成对比图与对比段落")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--env_note", default="- 后端：librosa.pyin（CREPE 未安装）")
    args = ap.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    overall = data["per_stage_overall"]
    stage_means = {s: overall[s]["mean"] for s in STAGES}
    fig_paths = [
        fig_dir / "fig1_stage_pie.png",
        fig_dir / "fig2_stage_bar.png",
        fig_dir / "fig3_case_total.png",
        fig_dir / "fig4_case_stack.png",
    ]
    fig_stage_pie(stage_means, fig_paths[0])
    fig_stage_bar(overall, fig_paths[1])
    fig_case_total(data["per_case"], fig_paths[2])
    fig_stage_stack(data["per_case"], fig_paths[3])

    baseline_data = None
    if args.baseline:
        baseline_data = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        cmp_path = fig_dir / "fig5_compare.png"
        fig_compare(baseline_data, data, cmp_path)
        fig_paths.append(cmp_path)

    write_report(data, fig_paths, out_dir / "latency_report.md",
                 args.env_note, baseline_data)
    print(f"report: {out_dir / 'latency_report.md'}")


if __name__ == "__main__":
    main()
