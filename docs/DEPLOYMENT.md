# HummingMusic 部署文档

> 维护：zx（week14）· 更新日期：2026-06-08 · 基线：`develop @ 353cabd`

本文覆盖本地开发、无显示器服务器（headless）、CI 三种部署形态，以及音色库 /
模型权重 / 环境变量等易踩坑项。README 面向使用者，本文面向部署者。

---

## 1. 依赖与前置

| 组件 | 版本 | 必需 | 说明 |
|---|---|---|---|
| Python | ≥ 3.10 | 是 | 开发用 3.10/3.12 均验证 |
| pip 依赖 | `requirements.txt` | 是 | 完整运行（含 torch/torchcrepe） |
| CUDA | ≥ 11.8 | 否 | 无 GPU 自动回退 CPU |
| FluidSynth | ≥ 2.0 | 否 | 高质量合成；缺失回退 pretty_midi 内置 |
| SoundFont (.sf2) | GM | FluidSynth 时需 | 见 §4 |

> CI 专用轻量依赖见 `requirements-ci.txt`（不装 torch/crepe，由
> `tests/conftest.py` mock）。**部署生产环境请用 `requirements.txt`**，
> 否则音高提取不可用。

## 2. 安装

```bash
git clone https://github.com/DntRain/HummingMusic.git
cd HummingMusic
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 模型权重放置

权重不入库（`models/**/*.pt|*.pth|*.npy` 已 gitignore），需单独获取：

```
models/quantizer/bilstm_crf.pt                    # 量化器 BiLSTM-CRF
models/style_transfer/vqvae.pt                    # 风格迁移 VQ-VAE
models/style_transfer/decoder_{pop,jazz,classical,folk}.pt  # 4 风格 decoder
models/style_transfer/{pop,jazz,classical,folk}_vector.npy  # 风格参考向量
```

- 缺任一权重时对应模块**自动回退**，pipeline 仍可端到端运行。
- ⚠️ **已知问题**：当前 `bilstm_crf.pt` 推理退化（不产 onset），真实哼唱
  产出 0 notes。临时兜底为**移走该权重强制 baseline**。详见
  `reports/week14/xyk/regression_report.md` 与 README「已知问题」。

## 4. SoundFont（FluidSynth 合成）

`config.yaml` 默认指向 `default.sf2`。两种配置方式：

```bash
# 方式一：安装系统 GM 音色库
sudo apt install fluid-soundfont-gm          # Ubuntu/Debian
sudo pacman -S soundfont-fluid               # Arch

# 方式二：指向已有 .sf2（如 FluidR3_GM.sf2），改 config.yaml:
#   renderer.soundfont: /usr/share/soundfonts/FluidR3_GM.sf2
```

> FluidSynth 2.5+ CLI 选项必须在 SF2 文件**之前**：
> `fluidsynth -ni -F out.wav -r 44100 <SF2> <MID>`（顺序错会报 illegal option）。
> 无 SoundFont 时系统回退 pretty_midi 内置合成，音质较低但不阻断。

## 5. 启动（本地）

```bash
streamlit run tools/visualizer.py --server.port 8501
# 浏览器访问 http://localhost:8501
```

## 6. 启动（headless 服务器）

```bash
export FLUID_NO_AUDIO_DRIVERS=1      # 无音频设备，禁止 fluidsynth 抢音频驱动
mkdir -p logs
nohup streamlit run tools/visualizer.py \
  --server.port 8501 --server.headless true \
  > logs/visualizer.log 2>&1 &
```

| 环境变量 | 作用 |
|---|---|
| `FLUID_NO_AUDIO_DRIVERS=1` | headless 下避免 fluidsynth 初始化音频驱动失败 |
| `CUDA_VISIBLE_DEVICES` | 指定/禁用 GPU（设空串强制 CPU） |

反向代理（nginx）需放行 WebSocket（Streamlit 依赖）：

```nginx
location / {
    proxy_pass http://127.0.0.1:8501;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

## 7. 部署前自检

```bash
pytest tests/ -q                 # 期望 93 passed, 1 xfailed
flake8 src/ tests/ --ignore=E501,W503 --count   # 期望 0
python -c "import torch, torchcrepe, librosa, pretty_midi; print('deps ok')"
```

> CI 已自动执行 flake8 + pytest（`.github/workflows/ci.yml`）。CI 全绿
> **不**覆盖模型层 e2e（CI 无模型、无 torch），生产部署须额外做真实音频冒烟。

## 8. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 真实哼唱产出 0 notes | Bug-A/Bug-B（见 README 已知问题） | 移走 `bilstm_crf.pt` 走 baseline |
| `ZeroDivisionError` BPM | 纯音/极短输入 | 已修（回退 120），升级到 `@353cabd`+ |
| fluidsynth `illegal option` | 2.5+ 参数顺序 | 选项放 SF2 之前 |
| `not a SoundFont` | SF2 路径错/缺失 | 见 §4，或接受 pretty_midi 回退 |
| CI flake8 失败 | lint | 本地先跑 §7 的 flake8 |

## 9. 版本与回归

- 回归测试套件：`tests/test_regression_week14.py`（RC-01~24）
- 回归记录：`reports/week14/{ly,zx}/regression_test_log.md`
- 根因分析：`reports/week14/xyk/regression_report.md`
