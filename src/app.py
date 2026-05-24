"""
app.py - Gradio 主入口

提供 Web 界面，包括：
- 录音/上传音频控件
- 风格选择
- 完整 pipeline 触发
- 结果展示（音频播放、Piano Roll、MIDI下载）
"""
import logging
import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import pretty_midi
import yaml
from matplotlib import font_manager
from matplotlib.patches import Patch

from src.interfaces import extract_pitch, quantize_humming, render_audio, transfer_style

matplotlib.use("Agg")

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _config = yaml.safe_load(_f)

logging.basicConfig(
    level=getattr(logging, _config["logging"]["level"]),
    format=_config["logging"]["format"],
)
logger = logging.getLogger(__name__)

for fp in [
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
]:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


STYLE_CHOICES = [
    ("🎵 流行 (pop)", "pop"),
    ("🎷 爵士 (jazz)", "jazz"),
    ("🎻 古典 (classical)", "classical"),
    ("🪕 民谣 (folk)", "folk"),
]

# Fix #7: 各阶段在 0-1 进度条上的真实权重，按 Week12 延迟基线测量结果（extract_pitch ~98%）
PROGRESS_WEIGHTS = {
    "extract": (0.00, 0.85),
    "quantize": (0.85, 0.88),
    "style": (0.88, 0.92),
    "render": (0.92, 0.97),
    "viz": (0.97, 1.00),
}


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _plot_piano_roll(midi: pretty_midi.PrettyMIDI) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    melody_color, accomp_color = "steelblue", "coral"

    for instrument in midi.instruments:
        color = melody_color if instrument.name == "melody" else accomp_color
        for note in instrument.notes:
            ax.barh(
                note.pitch, note.end - note.start,
                left=note.start, height=0.8, alpha=0.7, color=color,
            )

    # Fix #5: Piano Roll 添加图例
    ax.legend(
        handles=[
            Patch(facecolor=melody_color, alpha=0.7, label="旋律 (melody)"),
            Patch(facecolor=accomp_color, alpha=0.7, label="伴奏 (accompaniment)"),
        ],
        loc="upper right",
    )

    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("MIDI 音高")
    ax.set_title("Piano Roll")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    tmp_path = tempfile.mktemp(suffix=".png")
    fig.savefig(tmp_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return tmp_path


def _save_midi(midi: pretty_midi.PrettyMIDI) -> str:
    """Fix #1: 直接把 styled MIDI 写入唯一临时文件，供前端下载。

    原实现从 wav 路径推断 .mid，但 render_audio 不产 MIDI，导致下载永远为空。
    """
    out_dir = Path(tempfile.gettempdir()) / f"hum_{uuid.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_path = out_dir / "styled.mid"
    midi.write(str(midi_path))
    return str(midi_path)


def _friendly_error(exc: Exception) -> str:
    """Fix #9: 异常脱敏，给用户友好提示，详细 traceback 只写日志。"""
    logger.exception("Pipeline 失败: %s", exc)
    name = type(exc).__name__
    if isinstance(exc, FileNotFoundError):
        return "❌ 找不到音频文件，请重新录音或上传。"
    if isinstance(exc, ValueError):
        return f"❌ 输入参数不合法（{name}），请检查后重试。"
    return f"❌ 处理失败：{name}。已记录到服务端日志，请稍后重试或联系管理员。"


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

def process_humming(
    audio_input: str | None,
    style: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str | None, str | None, str]:
    if audio_input is None:
        return None, None, None, "⚠️ 请先在左侧录音或上传音频文件。"

    def _step(name: str, msg: str):
        lo, _ = PROGRESS_WEIGHTS[name]
        progress(lo, desc=msg)

    try:
        t0 = time.perf_counter()

        _step("extract", "🎤 正在提取音高...")
        pitch_data = extract_pitch(audio_input)
        t_extract = time.perf_counter() - t0
        status = (f"✓ 音高提取完成：{len(pitch_data['time'])} 帧 "
                  f"| BPM={pitch_data['bpm']:.1f} | {t_extract:.2f}s\n")

        _step("quantize", "🎼 正在量化为 MIDI...")
        midi_quantized = quantize_humming(pitch_data)
        n_notes = sum(len(inst.notes) for inst in midi_quantized.instruments)
        status += f"✓ 量化完成：{n_notes} 个音符\n"

        _step("style", f"🎨 正在迁移风格 ({style})...")
        midi_styled = transfer_style(midi_quantized, style)
        n_tracks = len(midi_styled.instruments)
        status += f"✓ 风格迁移完成：{n_tracks} 个轨道\n"

        _step("render", "🔊 正在渲染音频...")
        wav_path = render_audio(midi_styled)
        midi_path = _save_midi(midi_styled)
        status += "✓ 渲染完成\n"

        _step("viz", "🖼 正在生成可视化...")
        piano_roll_img = _plot_piano_roll(midi_styled)
        progress(1.0, desc="完成!")

        total = time.perf_counter() - t0
        status += f"\n🎉 全部完成，总耗时 {total:.2f}s"
        return wav_path, piano_roll_img, midi_path, status

    except Exception as exc:
        return None, None, None, _friendly_error(exc)


def _lock_button():
    """Fix #2: 提交时立刻禁用按钮，防止并发点击。"""
    return gr.update(value="⏳ 处理中…", interactive=False)


def _unlock_button():
    return gr.update(value="🎵 生成", interactive=True)


def _stale_warning(_audio, _style):
    """Fix #10: 风格变更时给出"结果已过期"提示。"""
    return "ℹ️ 已修改输入/风格，请点击「生成」重新处理。"


# ──────────────────────────────────────────────
# Gradio 界面
# ──────────────────────────────────────────────

def create_ui() -> gr.Blocks:
    with gr.Blocks(
        title="HummingMusic - 哼唱旋律风格迁移",
        theme=gr.themes.Soft(),
        css="footer {visibility: hidden}",
    ) as app:
        gr.Markdown(
            """
            # 🎶 HummingMusic — 哼唱旋律风格迁移
            录一段哼唱或上传音频，选择目标风格，生成风格化音乐。
            > 提示：如同时录制麦克风并上传文件，系统会**优先使用麦克风录音**。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎙️ 输入")
                audio_mic = gr.Audio(
                    sources=["microphone"], type="filepath",
                    label="麦克风录音（优先）",
                )
                audio_upload = gr.Audio(
                    sources=["upload"], type="filepath",
                    label="或上传音频 (WAV / MP3 / M4A)",
                )
                style_dropdown = gr.Dropdown(
                    choices=STYLE_CHOICES, value="pop",
                    label="🎨 目标风格",
                    info="共 4 种，决定生成结果的配器与节奏特征",
                )
                generate_btn = gr.Button(
                    "🎵 生成", variant="primary", size="lg"
                )

                gr.Markdown("#### 💡 示例")
                demo_root = Path(__file__).resolve().parent.parent / "data" / "demo"
                example_wav = demo_root / "example.wav"
                examples = []
                if example_wav.exists():
                    for s in ["pop", "jazz", "classical", "folk"]:
                        examples.append([str(example_wav), s])
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[audio_upload, style_dropdown],
                        label="点击任意示例载入",
                    )

            with gr.Column(scale=2):
                gr.Markdown("### 🎧 输出")
                output_audio = gr.Audio(
                    label="生成的音频", type="filepath", interactive=False,
                )
                piano_roll_img = gr.Image(
                    label="Piano Roll 可视化", type="filepath",
                )
                midi_download = gr.File(label="📥 下载 MIDI 文件")
                status_text = gr.Textbox(
                    label="处理状态", lines=10, max_lines=20,
                    interactive=False,
                )

        def _get_audio_input(mic_audio, upload_audio):
            return mic_audio if mic_audio is not None else upload_audio

        def _run(mic, upload, style):
            return process_humming(_get_audio_input(mic, upload), style)

        # Fix #2: lock → run → unlock 链式串行，并发被 queue 卡在 limit=1
        generate_btn.click(
            fn=_lock_button, inputs=None, outputs=generate_btn, queue=False
        ).then(
            fn=_run,
            inputs=[audio_mic, audio_upload, style_dropdown],
            outputs=[output_audio, piano_roll_img, midi_download, status_text],
        ).then(
            fn=_unlock_button, inputs=None, outputs=generate_btn, queue=False
        )

        # Fix #10: 输入变更时提示结果已过期
        for comp in (audio_mic, audio_upload, style_dropdown):
            comp.change(
                fn=_stale_warning, inputs=[audio_upload, style_dropdown],
                outputs=status_text, queue=False,
            )

    app.queue(default_concurrency_limit=1)
    return app


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
