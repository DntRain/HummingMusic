# HummingMusic

> 基于人声哼唱的旋律转译与风格迁移系统 —— 哼一段旋律，生成多风格编曲音频。

## 系统架构

```
┌─────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  哼唱    │───▶│  音频处理     │───▶│  容错量化     │───▶│  风格迁移     │───▶│  音频渲染 │
│  输入    │    │ (CREPE F0)   │    │(BiLSTM-CRF)  │    │  (VQ-VAE)    │    │(FluidSynth)│
└─────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘
  WAV/MP3        time, freq,         PrettyMIDI           PrettyMIDI          WAV
  录音/上传       confidence, bpm     单轨旋律              旋律+伴奏           可播放音频
```

**支持四种目标风格：** Pop (流行) · Jazz (爵士) · Classical (古典) · Folk (民谣)

## 环境要求

| 依赖项 | 版本要求 |
|--------|---------|
| Python | >= 3.10 |
| CUDA   | >= 11.8（可选，GPU 加速推理） |
| FluidSynth | >= 2.0（可选，高质量合成） |

> 无 GPU 和 FluidSynth 也可运行，系统会自动 fallback 到 CPU 推理和 pretty_midi 内置合成。

## 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/<your-org>/HummingMusic.git
cd HummingMusic

# 2. 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4.（可选）安装 FluidSynth 和音色库
# Ubuntu/Debian:
sudo apt install fluidsynth fluid-soundfont-gm
# Arch Linux:
sudo pacman -S fluidsynth soundfont-fluid
```

## 快速开始

```bash
# 启动 Gradio Web 界面
python -m src.app
```

浏览器访问 `http://localhost:7860`，录音或上传哼唱音频，选择风格，点击"生成"。

## 数据集准备

本项目使用以下公开数据集，请自行下载并放入对应目录：

| 数据集 | 用途 | 下载地址 | 存放目录 |
|--------|------|---------|---------|
| HumTrans | 哼唱-MIDI 配对训练 | [HumTrans GitHub](https://github.com/shansongliu/HumTrans) | `data/humming/` |
| MIR-QBSH | 哼唱查询评测 | [MIR-QBSH](http://www.music-ir.org/mirex/wiki/2023:Query_by_Singing/Humming) | `data/humming/` |
| Lakh MIDI Dataset | 各风格MIDI参考 | [Lakh MIDI](https://colinraffel.com/projects/lmd/) | `data/midi_*/` |

数据集划分索引放在 `data/splits/` 目录下（train.txt / val.txt / test.txt）。

## 模型训练

> 训练脚本正在开发中，后续补充。

预训练权重放置路径：
- 量化器 (BiLSTM-CRF): `models/quantizer/bilstm_crf.pt`
- 风格迁移 (VQ-VAE): `models/style_transfer/vqvae.pt`
- 风格向量: `models/style_transfer/{pop,jazz,classical,folk}_vector.npy`

## 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 带覆盖率报告
pytest tests/ -v --cov=src --cov-report=term-missing
```

## 项目结构

```
HummingMusic/
├── src/
│   ├── interfaces.py           # 模块间接口定义（4个核心接口）
│   ├── audio_processing.py     # 音频处理：CREPE F0 提取 + 插值修复
│   ├── quantizer.py            # 容错量化：BiLSTM-CRF 序列标注
│   ├── style_transfer.py       # 风格迁移：VQ-VAE + music21 和弦推断
│   ├── renderer.py             # 音频渲染：FluidSynth / pretty_midi
│   └── app.py                  # Gradio Web 界面入口
├── data/
│   ├── humming/                # 哼唱音频数据
│   ├── midi_pop/               # 流行风格 MIDI 参考库
│   ├── midi_jazz/              # 爵士风格 MIDI 参考库
│   ├── midi_classical/         # 古典风格 MIDI 参考库
│   ├── midi_folk/              # 民谣风格 MIDI 参考库
│   └── splits/                 # 训练/验证/测试集划分索引
├── models/
│   ├── quantizer/              # BiLSTM-CRF 模型权重
│   └── style_transfer/         # VQ-VAE 模型权重 + 风格向量
├── experiments/                # 实验日志和记录
├── tmp/                        # 临时文件（运行时自动创建 UUID 子目录）
├── tests/                      # pytest 单元测试
├── .github/workflows/ci.yml   # GitHub Actions CI 配置
├── config.yaml                 # 统一配置文件
├── requirements.txt            # Python 依赖
├── CONTRIBUTING.md             # 协作规范
└── README.md
```

## 架构设计要点

- **接口隔离**：模块间严格通过 `interfaces.py` 定义的 4 个接口通信
- **优雅降级**：每个模块在模型权重未加载时有 fallback，pipeline 始终可端到端运行
- **并发安全**：`tmp/` 目录下每次请求使用 UUID 命名独立子目录
- **配置集中**：所有路径和超参数通过 `config.yaml` 管理，代码无硬编码

## 团队成员

| 角色 | 姓名 | 负责模块 |
|------|------|---------|
| 成员A | 【成员1】 | audio_processing（音频处理） |
| 成员B | 【成员2】 | quantizer（容错量化） |
| 成员C | 【成员3】 | style_transfer（风格迁移） |
| 成员D | 【成员4】 | app / renderer（界面与渲染） |

## 参考文献

1. Kim, J. W., et al. "CREPE: A Convolutional Representation for Pitch Estimation." *ICASSP*, 2018.
2. Huang, Z., Xu, W., & Yu, K. "Bidirectional LSTM-CRF Models for Sequence Tagging." *arXiv:1508.01991*, 2015.
3. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. "Neural Discrete Representation Learning (VQ-VAE)." *NeurIPS*, 2017.
4. Dong, H.-W., et al. "Music Transformer: Generating Music with Long-Term Structure." *ICLR*, 2019.
5. Raffel, C. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching." *PhD Thesis, Columbia University*, 2016.
6. Cuthbert, M. S. & Ariza, C. "music21: A Toolkit for Computer-Aided Musicology." *ISMIR*, 2010.
7. Liu, S., et al. "HumTrans: A Dataset and Benchmark for Humming Melody Transcription." *ISMIR*, 2022.

## License

MIT
