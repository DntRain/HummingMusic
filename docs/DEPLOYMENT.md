# HummingMusic 部署文档

本文档用于第十四周部署交付，说明如何从空环境配置 HummingMusic，并启动 Streamlit 前端完成基础验证。部署目标是让演示人员或测试人员能够按步骤复现系统运行环境，而不需要阅读源码。HummingMusic 的主入口是 `tools/visualizer.py`，默认端口为 `8501`，浏览器访问 `http://localhost:8501` 即可使用录音、上传音频、查看钢琴卷帘和试听风格迁移结果。

## 1. 环境要求

推荐使用 Python `3.10` 或更高版本。项目依赖写在 `requirements.txt` 中，主要包括 `streamlit`、`librosa`、`pretty_midi`、`music21`、`torch`、`crepe`、`matplotlib`、`numpy`、`scipy` 和 `soundfile`。其中 `torch` 和 `crepe` 用于模型推理和音高提取，`streamlit` 用于前端页面，`pretty_midi` 和 `pyfluidsynth` 用于 MIDI 与音频合成。GPU 不是强制要求；没有 GPU 时系统可以使用 CPU。FluidSynth 与 SoundFont 也不是强制要求；缺少音色库时渲染模块会退回到 `pretty_midi` 内置合成，音色质量会降低，但基本功能可以继续验证。

部署前建议确认以下条件：

- 本机可以执行 `python --version` 或 `python3 --version`。
- 项目根目录包含 `requirements.txt`、`config.yaml` 和 `tools/visualizer.py`。
- 当前 shell 对项目目录有读写权限，因为系统会在 `tmp/` 和 `data/recordings/` 下写入临时输出或录音文件。
- 如果需要正式展示效果，应提前准备模型权重和 SoundFont。

## 2. Windows PowerShell 部署步骤

进入项目目录：

```powershell
cd D:\Desktop\小组作业\music_hum\HummingMusic
```

创建并使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

启动前端：

```powershell
.\.venv\Scripts\python.exe -m streamlit run tools\visualizer.py --server.port 8501
```

如果端口 `8501` 被占用，可以改用其他端口，例如：

```powershell
.\.venv\Scripts\python.exe -m streamlit run tools\visualizer.py --server.port 8502
```

## 3. Linux / macOS 部署步骤

进入项目目录：

```bash
cd /path/to/HummingMusic
```

创建并激活虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

启动前端：

```bash
python -m streamlit run tools/visualizer.py --server.port 8501
```

如果部署在服务器上并希望后台运行，可以使用如下形式：

```bash
nohup python -m streamlit run tools/visualizer.py --server.port 8501 --server.headless true > visualizer.log 2>&1 &
```

## 4. 可选音色库配置

高质量音频合成建议安装 FluidSynth 和 SoundFont。Linux 上可以执行：

```bash
sudo apt install fluidsynth fluid-soundfont-gm
```

`config.yaml` 中的默认音色路径为：

```yaml
renderer:
  soundfont_path: "/usr/share/soundfonts/default.sf2"
```

如果系统音色库路径不同，请将 `soundfont_path` 改为实际 `.sf2` 文件路径。Windows 用户可以下载 GM SoundFont 文件后放入固定目录，再在 `config.yaml` 中填写绝对路径。没有 SoundFont 时不需要阻塞部署，系统仍可走 fallback 合成，但正式演示前建议补齐。

## 5. 模型权重放置

核心模块读取 `config.yaml` 中的模型路径。默认放置位置如下：

| 模型或资源 | 路径 |
|---|---|
| 量化器权重 | `models/quantizer/bilstm_crf.pt` |
| Streamlit 可视化量化器权重 | `models/quantizer_v5/bilstm_crf.pt` |
| VQ-VAE 权重 | `models/style_transfer/vqvae.pt` |
| 风格 decoder | `models/style_transfer/decoder_{pop,jazz,classical,folk}.pt` |
| 风格向量 fallback | `models/style_transfer/{pop,jazz,classical,folk}_vector.npy` |

如果没有完整权重，系统会尽量使用 baseline 或 fallback 路径。此时可以验证页面、文件输入、MIDI 输出和 WAV 渲染稳定性，但模型效果不代表正式版本。正式交付时，应将权重复制到上述目录，并重新运行测试。

## 6. 启动后验证

启动成功后，终端会显示本地访问地址。打开浏览器访问：

```text
http://localhost:8501
```

推荐按以下顺序验证：

1. 使用内置 demo 样本，确认页面能加载并显示推理结果。
2. 上传一段 `wav`、`mp3`、`m4a`、`flac` 或 `ogg` 音频，确认系统可以提取音高。
3. 查看钢琴卷帘页面，确认预测 MIDI 有可视化结果。
4. 切换到风格迁移页面，确认 pop、jazz、classical、folk 至少一种风格可以生成音频。
5. 运行单元测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Linux / macOS 使用：

```bash
python -m pytest tests/ -q
```

## 7. 常见问题

如果 `pip install -r requirements.txt` 失败，先升级 `pip`，并确认 Python 版本满足要求。`torch` 或 `crepe` 安装失败时，可以先完成其他依赖安装，再根据本机 CPU/GPU 环境选择合适版本。

如果前端启动后页面无法访问，优先检查端口是否被占用，并确认启动命令中的端口和浏览器访问端口一致。服务器部署时还需要确认防火墙或安全组允许访问对应端口。

如果音频输出为空或音色较差，通常是 SoundFont 缺失或 FluidSynth 不可用。此时功能流程仍可能通过，但正式展示前应安装 `.sf2` 音色库并更新 `config.yaml`。

如果风格迁移效果不明显，检查 `models/style_transfer/` 下是否存在 VQ-VAE、decoder 和风格向量文件。缺少权重时系统会走 fallback，稳定性可验证，但音乐效果会受限制。

如果数据集模式无法读取样本，说明本机没有对应外部数据集目录。演示部署优先使用 demo 或上传模式；需要数据集评测时，再按项目指南准备 HumTrans 等数据。
