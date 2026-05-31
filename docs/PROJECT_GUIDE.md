# HummingMusic 项目指南

> 面向新加入的队友：读完这份文档你应该能（1）理解整个系统是怎么从「一段哼唱」变成「四种风格的伴奏音频」的，（2）独立修改 `src/` 下任一模块而不破坏整条流水线，（3）把可视化前端 `tools/visualizer.py` 跑起来并按需扩展。

---

## 1. 系统总览

### 1.1 端到端数据流

```
WAV/MP3 ─► CREPE F0 ─► BiLSTM-CRF ─► VQ-VAE + decoder_<style> ─► style_postprocess ─► FluidSynth ─► WAV
  音频       (T, 4)      PrettyMIDI         48×T piano-roll           多轨 MIDI            playable
            音高特征      单轨 melody         按风格 decode             melody+和弦+鼓
```

四个核心阶段全部走 `src/interfaces.py` 暴露的 4 个顶层函数：

| 阶段 | 入口 | 实现位置 | 产物 |
|---|---|---|---|
| 音高提取 | `extract_pitch(audio_path)` | `src/audio_processing.py` | `dict(time, frequency, confidence, bpm)` |
| 容错量化 | `quantize_humming(pitch_data)` | `src/quantizer.py` | 单轨 `PrettyMIDI` |
| 风格迁移 | `transfer_style(midi, style)` | `src/style_transfer.py` | 含旋律的 `PrettyMIDI` |
| 音频渲染 | `render_audio(midi)` | `src/renderer.py` | WAV 路径 |

**为什么走接口？** 任何模块的内部实现都可以重写，只要保证签名和返回字段不变，其它模块、可视化工具、测试都不用动。修改前先看 `interfaces.py` 里那个 docstring 是契约。

### 1.2 目录速查

```
HummingMusic/
├── src/                       # 4 个核心模块 + 接口 + 风格后处理
│   ├── interfaces.py          # ⭐ 模块间唯一契约
│   ├── audio_processing.py    # CREPE / pyin + 插值修复
│   ├── quantizer.py           # BiLSTM-CRF 序列标注 + baseline 量化
│   ├── style_transfer.py      # VQ-VAE encode → 4 个 decoder
│   ├── style_postprocess.py   # 调性估计 + 风格化和弦/鼓伴奏
│   └── renderer.py            # FluidSynth / pretty_midi fallback
├── train/                     # 训练/评测脚本（CI 不依赖）
│   ├── extract_features.py    # 离线批量出 CREPE 特征 (T, 4)
│   ├── train.py               # BiLSTM-CRF 训练
│   ├── train_vqvae.py         # VQ-VAE 训练
│   ├── finetune_decoder.py    # 4 个 decoder 风格 fine-tune
│   ├── dataset.py             # HumTransDataset
│   ├── metrics.py             # note-level P/R/F1
│   └── evaluate.py / ablation.py / compare_methods.py
├── tools/                     # 一次性脚本 + 可视化
│   ├── visualizer.py          # ⭐ Streamlit 前端（主入口）
│   ├── bench_latency.py       # 端到端延迟测试
│   ├── extract_style_vectors.py
│   └── gen_*.py               # 图表、PPT、对比图生成
├── models/
│   ├── quantizer_v5/bilstm_crf.pt
│   └── style_transfer/
│       ├── vqvae.pt
│       ├── decoder_{pop,jazz,classical,folk}.pt   # ⭐ 4 个风格 decoder
│       └── *_vector.npy                            # 回退用的风格向量
├── config.yaml                # ⭐ 所有路径和超参数集中地
├── data/
│   ├── demo/                  # 演示音频（visualizer 默认样本）
│   ├── recordings/            # 现场录音落盘（首次录音时自动建）
│   ├── splits/                # train/valid/test 索引
│   └── midi_{pop,jazz,classical,folk}/  # 风格参考库
├── reports/week*/             # 周报、UI 迭代日志、bug fix 日志（.gitignore）
└── tests/                     # pytest，CI 跑这个
```

### 1.3 全局配置

`config.yaml` 是唯一的真相源，所有模块通过 `yaml.safe_load` 读取，**别在代码里硬编码路径或超参数**。关键字段：

- `quantizer.confidence_threshold: 0.5`：CREPE 帧置信度阈值。哼唱信号噪声大，调到 0.8 会滤掉约 90%，目前的 0.5 是经验值。
- `style_transfer.frame_rate: 32`：piano-roll 采样率（fps）。VQ-VAE 训练就是 32 fps，**改这里会破坏推理一致性**。
- `style_transfer.style_vectors_dir`：`decoder_<style>.pt` 和 `<style>_vector.npy` 都放这里。
- `renderer.soundfont_path`：合成音色库。系统没装 sf2 时 visualizer 会自己尝试一组 fallback 路径。

---

## 2. 各模块怎么改

### 2.1 `src/audio_processing.py`（音高提取）

- 输入：音频文件路径。
- 输出：`{time, frequency, confidence, bpm}` 四字段 dict。
- 关键点：CREPE 必须输出 `np.nan` 表示低置信度帧，下游 quantizer 依赖这个。
- 想换提取器（比如 SPICE）：保持返回 dict 结构，shape 一致即可。

### 2.2 `src/quantizer.py`（容错量化）

- 输入：`extract_pitch` 的 dict。
- 输出：单轨 `PrettyMIDI`（轨名通常是 `melody`）。
- 主路径：BiLSTM-CRF 做 BIO 序列标注，把帧级音高转成离散音符；权重在 `models/quantizer_v5/bilstm_crf.pt`。
- Fallback：模型未加载时走中位数 + silence 阈值的 baseline。
- **改训练**：去 `train/train.py`；改 ckpt 结构后注意 `_load_model` 兼容 `{model_state_dict, ...}` 嵌套格式。

### 2.3 `src/style_transfer.py`（风格迁移）

- 输入：`PrettyMIDI` + 目标风格字符串。
- 输出：`PrettyMIDI`（只含 melody 轨；伴奏由 `style_postprocess` 外加）。
- 主路径：
  1. `midi.get_piano_roll(fs=32)` → `(128, T)`，截 C2–C6 段 → `(48, T)`
  2. `vqvae.encode(x)` → 离散 code
  3. `decoder_<style>(z_q)` → 重建 48×T，sigmoid 概率 > 0.5 二值化、`(prob*127)` 当 velocity
  4. `frame_dur = 1.0 / frame_rate`，按帧 group 出 `Note(velocity, pitch, start, end)`
- Fallback：缺 decoder 时退回 `vqvae.decode(z_q + style_vec)` 单 decoder + 注入向量。
- **新增风格**：训练一个新的 `decoder_<name>.pt` 放到 `models/style_transfer/`，并把 `<name>` 加进 `interfaces.VALID_STYLES`。

### 2.4 `src/style_postprocess.py`（伴奏 + 调性）

- 给 melody-only MIDI 加和弦、鼓和踩镲。
- `stylize(midi, style, tempo=...)` 是对外入口；visualizer 里把情感缩放后的 bpm 传进来。
- Krumhansl-Schmuckler 估调性，按风格挑和弦进行表。

### 2.5 `src/renderer.py`（音频渲染）

- 优先 FluidSynth + 系统 sf2；失败时退到 `pretty_midi.fluidsynth()` 内置。
- 输出 WAV 路径在 `tmp/<uuid>/`，每次请求独立子目录，并发安全。

---

## 3. 可视化工具 `tools/visualizer.py`

### 3.1 启动

```bash
# 前台
streamlit run tools/visualizer.py --server.port 8501

# 后台 + 日志
nohup streamlit run tools/visualizer.py --server.port 8501 --server.headless true \
    > logs/visualizer.log 2>&1 &
```

浏览器开 `http://localhost:8501`，左侧切换 5 个页面：

| 页面 | 内容 |
|---|---|
| 🎤 输入与推理 | 指标卡 + 原始/预测/GT 三栏音频对比 |
| 🎹 钢琴卷帘 | 预测 vs GT piano-roll |
| 📊 模型得分 | BiLSTM-CRF 帧级 score 曲线 + 检出的峰值 |
| 📋 音符列表 | 预测音符的 (start, end, pitch, velocity) 表 |
| 🎨 风格迁移 | 4 风格 × 5 情感的合成结果 |

### 3.2 三种输入源

按优先级从高到低：

1. **现场录音**（`st.audio_input`）：先落盘到 `data/recordings/rec_<md5前10位>.<ext>`，再当作上传文件走。同一段内容多次录不会产生重复文件（md5 去重）。
2. **上传文件**（`st.file_uploader`）：支持 wav/mp3/flac/ogg/m4a，自动 ffmpeg 转 16k mono。
3. **数据集模式**：从 `HumTransDataset` 拿样本，配 GT 算指标；侧边栏选 split，主区域 slider + `‹ ›` 按钮翻页。

切换演示样本走 `data/demo/example.m4a`。

### 3.3 三层缓存（理解它，性能问题就好查了）

| 缓存 | 装饰器 | 键 | 作用 |
|---|---|---|---|
| 模型常驻 | `@st.cache_resource` | 无 | `load_model` / `load_vqvae` / `load_dataset` 首次后内存常驻 |
| 函数输入哈希 | `@st.cache_data` | 函数参数 | `extract_features_from_bytes` / `synthesize_midi` / `run_style_transfer` 相同字节直接复用 |
| 推理指纹 | `st.session_state["infer_fp"]` | `(input_fp, peak_distance, peak_height_sigma)` | 切页、改无关参数不重跑推理 |

**强制重跑**：侧边栏底部「🔄 清空推理结果」会清掉 `confirmed_input_fp / last_result / infer_fp / ds_idx`。

### 3.4 情感调制（A 方案）

`EMOTION_PARAMS`（约 472–479 行）定义 5 档情感的四元组：

```python
{"bpm_scale": 1.10, "vel_scale": 1.05, "vel_offset": +8, "vel_range": 1.15}  # happy
```

- `bpm_scale`：传给 `stylize` 影响和弦/鼓时长。
- `vel_range`：以 64 为中心做对称压扩，控制力度动态范围。
- `vel_scale` × velocity + `vel_offset`：仿射变换整体力度。
- 加新情感 → 加一行 `EMOTION_PARAMS`，再到 `PAGE_STYLE` 的 radio 里加选项；缓存键已经包含 `emotion`，不会撞老结果。

### 3.5 怎么改 visualizer

不同改动的入口锚点：

| 想做的事 | 改哪里 |
|---|---|
| 加一个新页面 | `ALL_PAGES`（约 621 行）加常量；底部 `if page == ...` 加分发分支 |
| 加一个侧边栏控件 | 选合适的 `st.expander`（约 638–671 行）插入；记得把它的值并入 `infer_fp` 否则缓存不会失效 |
| 改推理参数默认值 | `st.slider` 的第 4 个位置参数（`peak_distance` 约 656 行） |
| 加一个合成音色 | `instrument_name = st.selectbox(...)` 的 list 里加 GM 名 |
| 换 BiLSTM-CRF 模型 | 改 `CKPT_PATH`（约 41 行）；ckpt 结构变了同步改 `load_model` |
| 加新数据集 | 改 `DATA_ROOT / SPLIT_JSON / WAV_DIR / MIDI_DIR`（约 39–44 行），保证 `HumTransDataset` 能读 |
| 加 style fallback 路径 | `_SF2_CANDIDATES`（约 46–52 行）追加候选 sf2 |

**改完后**：Streamlit 会自动 reload，但 `@st.cache_resource` 不会自动失效；改了 `load_model` / `load_vqvae` 要么改函数签名、要么手动 `st.cache_resource.clear()`。

### 3.6 常见排错

| 现象 | 检查项 |
|---|---|
| `weights_only=True` 加载失败 | ckpt 是嵌套结构，需要 `state.get("model_state_dict", state)` |
| 风格迁移区一直 spinner | 看 `models/style_transfer/decoder_<style>.pt` 是否存在；缺会走 fallback 路径但不会 hang |
| 录音上传后没反应 | 看 `data/recordings/` 是否生成 `rec_*.wav`，没生成说明 `_save_recording_to_disk` 抛了异常 |
| 数据集模式很卡 | TRAIN split 13080 条首次加载 8–12s，已 `@st.cache_resource` 后秒返；如果反复重载说明 ckpt 路径变了导致 cache key 改 |
| 钢琴卷帘是空的 | 看面包屑「预测音符数」；为 0 时通常是 CREPE 置信度过低（哼太轻、噪声大） |

---

## 4. 开发约定

1. **先看 `interfaces.py` 再动 `src/`**：契约比实现重要。
2. **配置进 `config.yaml`**：代码里不要写绝对路径。
3. **改 ckpt 结构同步改 loader**：兼容 `dict[model_state_dict]` 和裸 `state_dict` 两种形态。
4. **可视化改动配套日志**：本周的 UI 迭代写在 `reports/week<n>/dc/ui_iteration_log.md`（被 .gitignore）。
5. **commit 前先跑测试**：`pytest tests/ -v`，CI 配置见 `.github/workflows/`。
6. **commit 信息格式**：`[<scope>] <type>(<area>): <subject>`，例 `[week12] feat(ui): 五页改造`。
