"""
app.py - Gradio 主入口

提供 Web 界面，包括：
- 录音/上传音频控件
- 风格选择
- 完整 pipeline 触发
- 结果展示（音频播放、Piano Roll、MIDI下载）
"""
#test
import logging
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import yaml

# 使用非交互后端
matplotlib.use("Agg")

# 加载配置
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _config = yaml.safe_load(_f)

# 配置日志
logging.basicConfig(
    level=getattr(logging, _config["logging"]["level"]),
    format=_config["logging"]["format"],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 通过 interfaces 调用各模块（严格遵守接口约定）
# ──────────────────────────────────────────────

from src.interfaces import extract_pitch, quantize_humming, render_audio, transfer_style


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _plot_piano_roll(midi: pretty_midi.PrettyMIDI) -> str:
    """
    绘制 Piano Roll 可视化图像。

    Args:
        midi: PrettyMIDI 对象。

    Returns:
        str: 图像文件路径。
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    for instrument in midi.instruments:
        for note in instrument.notes:
            ax.barh(
                note.pitch,
                note.end - note.start,
                left=note.start,
                height=0.8,
                alpha=0.7,
                color="steelblue" if instrument.name == "melody" else "coral",
            )

    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("MIDI 音高")
    ax.set_title("Piano Roll")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存到临时文件
    tmp_path = tempfile.mktemp(suffix=".png")
    fig.savefig(tmp_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return tmp_path


def _get_midi_path(wav_path: str) -> str | None:
    """
    从 WAV 路径推断同目录下的 MIDI 文件路径。

    Args:
        wav_path: WAV 文件路径。

    Returns:
        str | None: MIDI 文件路径，不存在则返回 None。
    """
    midi_path = str(Path(wav_path).with_suffix(".mid"))
    if Path(midi_path).exists():
        return midi_path
    return None


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

def process_humming(
    audio_input: str | None,
    style: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str | None, str | None, str]:
    """
    完整处理流水线：音高提取 → 量化 → 风格迁移 → 渲染。

    Args:
        audio_input: 输入音频文件路径（来自录音或上传）。
        style: 目标风格。
        progress: Gradio 进度条。

    Returns:
        tuple: (wav_path, piano_roll_image_path, midi_path, status_message)
    """
    if audio_input is None:
        return None, None, None, "请先录音或上传音频文件。"

    try:
        # 阶段1：音高提取
        progress(0.1, desc="正在提取音高...")
        logger.info("Pipeline 开始: 音高提取")
        pitch_data = extract_pitch(audio_input)
        status = f"音高提取完成: {len(pitch_data['time'])} 帧, BPM={pitch_data['bpm']:.1f}\n"

        # 阶段2：量化
        progress(0.3, desc="正在量化为MIDI...")
        logger.info("Pipeline: 量化")
        midi_quantized = quantize_humming(pitch_data)
        n_notes = sum(len(inst.notes) for inst in midi_quantized.instruments)
        status += f"量化完成: {n_notes} 个音符\n"

        # 阶段3：风格迁移
        progress(0.5, desc=f"正在进行风格迁移 ({style})...")
        logger.info("Pipeline: 风格迁移 → %s", style)
        midi_styled = transfer_style(midi_quantized, style)
        n_tracks = len(midi_styled.instruments)
        status += f"风格迁移完成: {n_tracks} 个轨道\n"

        # 阶段4：渲染
        progress(0.8, desc="正在渲染音频...")
        logger.info("Pipeline: 渲染")
        wav_path = render_audio(midi_styled)
        status += f"渲染完成: {wav_path}\n"

        # 生成可视化
        progress(0.9, desc="正在生成可视化...")
        piano_roll_img = _plot_piano_roll(midi_styled)

        # 获取MIDI文件路径
        midi_path = _get_midi_path(wav_path)

        progress(1.0, desc="完成!")
        status += "全部处理完成！"
        logger.info("Pipeline 完成")

        return wav_path, piano_roll_img, midi_path, status

    except FileNotFoundError as e:
        msg = f"文件未找到: {e}"
        logger.error(msg)
        return None, None, None, msg
    except ValueError as e:
        msg = f"参数错误: {e}"
        logger.error(msg)
        return None, None, None, msg
    except Exception as e:
        msg = f"处理失败: {e}"
        logger.exception(msg)
        return None, None, None, msg


# ──────────────────────────────────────────────
# Gradio 界面
# ──────────────────────────────────────────────

def create_ui() -> gr.Blocks:
    """
    创建 Gradio Blocks 界面。

    Returns:
        gr.Blocks: Gradio 应用实例。
    """
    with gr.Blocks(
        title="HummingMusic - 哼唱旋律风格迁移",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # HummingMusic - 哼唱旋律风格迁移系统
            哼唱一段旋律，选择目标风格，系统将自动转译并生成风格化音频。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 输入")

                audio_mic = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="麦克风录音",
                )
                audio_upload = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="上传音频 (WAV/MP3)",
                )

                style_dropdown = gr.Dropdown(
                    choices=["pop", "jazz", "classical", "folk"],
                    value="pop",
                    label="目标风格",
                )

                generate_btn = gr.Button(
                    "生成", variant="primary", size="lg"
                )

            with gr.Column(scale=2):
                gr.Markdown("### 输出")

                output_audio = gr.Audio(
                    label="生成的音频",
                    type="filepath",
                    interactive=False,
                )

                piano_roll_img = gr.Image(
                    label="Piano Roll 可视化",
                    type="filepath",
                )

                midi_download = gr.File(
                    label="下载 MIDI 文件",
                )

                status_text = gr.Textbox(
                    label="处理状态",
                    lines=5,
                    interactive=False,
                )

        # 事件绑定：优先使用麦克风录音，其次使用上传文件
        def _get_audio_input(mic_audio, upload_audio):
            return mic_audio if mic_audio is not None else upload_audio

        generate_btn.click(
            fn=lambda mic, upload, style: process_humming(
                _get_audio_input(mic, upload), style
            ),
            inputs=[audio_mic, audio_upload, style_dropdown],
            outputs=[output_audio, piano_roll_img, midi_download, status_text],
        )

    return app


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
