# HummingMusic

> 基于人声哼唱的旋律转译与风格迁移系统。用户录制或上传一段哼唱音频后，系统完成音高提取、旋律量化、风格迁移和音频渲染，并在 Streamlit 前端中展示音频、MIDI、钢琴卷帘和多风格结果。

## 系统架构

```text
哼唱输入
  -> 音频处理 extract_pitch(audio_path)
  -> 容错量化 quantize_humming(pitch_data)
  -> 风格迁移 transfer_style(midi, style)
  -> 音频渲染 render_audio(midi)
  -> Streamlit 前端展示与试听
```

支持的目标风格：

- Pop / 流行
- Jazz / 爵士
- Classical / 古典
- Folk / 民谣

## 快速开始

详细部署步骤见 [部署文档](docs/DEPLOYMENT.md)。本地已有 Python 环境时，可以按下面步骤启动。

### Windows PowerShell

```powershell
cd HummingMusic
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run tools\visualizer.py --server.port 8501
```

### Linux / macOS

```bash
cd HummingMusic
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run tools/visualizer.py --server.port 8501
```

启动后访问：

```text
http://localhost:8501
```

## 环境与资源

| 项目 | 要求 |
|---|---|
| Python | >= 3.10 |
| Python 依赖 | `requirements.txt` |
| 前端入口 | `tools/visualizer.py` |
| 默认端口 | `8501` |
| 配置文件 | `config.yaml` |
| 临时输出目录 | `tmp/` |

GPU、完整模型权重、FluidSynth 和 SoundFont 都是可选增强项。缺少这些资源时，核心模块会尽量走 CPU 或 fallback 路径，保证基本流程仍可运行；但正式展示时建议补齐模型权重和音色库，以获得更好的识别与合成效果。

模型和资源路径：

| 资源 | 默认位置 |
|---|---|
| 量化器权重 | `models/quantizer/bilstm_crf.pt` |
| Streamlit 量化器权重 | `models/quantizer_v5/bilstm_crf.pt` |
| VQ-VAE 权重 | `models/style_transfer/vqvae.pt` |
| 风格 decoder | `models/style_transfer/decoder_{pop,jazz,classical,folk}.pt` |
| 风格向量 | `models/style_transfer/{pop,jazz,classical,folk}_vector.npy` |
| 演示音频 | `data/demo/` |
| 录音落盘目录 | `data/recordings/` |

## 运行测试

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

```bash
# Linux / macOS
python -m pytest tests/ -q
```

第十三周回归脚本可用于稳定性验收：

```powershell
.\.venv\Scripts\python.exe tools\week13_regression.py
```

## 项目结构

```text
HummingMusic/
├── src/
│   ├── interfaces.py
│   ├── audio_processing.py
│   ├── quantizer.py
│   ├── style_transfer.py
│   ├── style_postprocess.py
│   └── renderer.py
├── tools/
│   ├── visualizer.py
│   ├── week13_regression.py
│   └── week13_json_figures.py
├── train/
├── tests/
├── data/
├── models/
├── docs/
│   ├── DEPLOYMENT.md
│   └── PROJECT_GUIDE.md
├── config.yaml
├── requirements.txt
└── README.md
```

## 设计要点

- **接口隔离**：模块间通过 `src/interfaces.py` 统一调用。
- **优雅降级**：缺少部分模型或音色库时，系统会使用 fallback 路径。
- **前端统一**：当前主入口为 Streamlit，不再依赖旧 Gradio 入口。
- **配置集中**：采样率、模型路径、SoundFont 路径和临时目录都在 `config.yaml` 中维护。
- **输出隔离**：渲染结果写入 `tmp/<uuid>/`，避免多次请求互相覆盖。

## 相关文档

- [部署文档](docs/DEPLOYMENT.md)
- [项目指南](docs/PROJECT_GUIDE.md)

## License

MIT
