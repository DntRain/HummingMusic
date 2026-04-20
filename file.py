import gradio as gr
import os


# 后端处理逻辑
def process_audio(audio_path, style):
    if audio_path is None:
        return None, "❌ 未检测到文件，请上传 .wav 音频。"

    # 获取文件名，用于日志展示
    file_name = os.path.basename(audio_path)
    log = f"已成功读取文件：{file_name}\n选择风格：{style}\n状态：准备进行第3周和弦推断。"

    # 返回原路径用于播放测试
    return audio_path, log


# 界面自定义样式
custom_css = """
.container { max-width: 900px; margin: auto; }
.title { text-align: center; font-family: 'Arial'; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🎶 智能音乐生成系统", elem_classes="title")
    gr.Markdown("### 阶段：第 2 周 - UI 交互与文件上传测试")

    with gr.Row():
        # 输入区
        with gr.Column():
            gr.Markdown("#### 📥 输入部分")
            # 核心组件：支持拖拽和点击上传
            input_audio = gr.Audio(
                label="上传你的 WAV 旋律文件",
                sources=["upload", "microphone"],  # 双源输入
                type="filepath",
                interactive=True  # 确保组件是可交互的
            )

            style_dropdown = gr.Dropdown(
                choices=["流行", "古典", "爵士", "电子"],
                value="流行",
                label="选择伴奏风格"
            )

            run_btn = gr.Button("开始处理", variant="primary")

        # 输出区
        with gr.Column():
            gr.Markdown("#### 📤 输出部分")
            output_audio = gr.Audio(label="音频预览", interactive=False)
            status_box = gr.Textbox(label="处理日志", lines=5)

    # 绑定事件
    run_btn.click(
        fn=process_audio,
        inputs=[input_audio, style_dropdown],
        outputs=[output_audio, status_box]
    )

if __name__ == "__main__":
    # 启动界面
    demo.launch(
        auth=("dou_chang", "music123"),  # 你的本地账号
        server_name="127.0.0.1"
    )