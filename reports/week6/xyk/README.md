# 第六周：对比实验报告

## 任务目标

3 种方法在 HumTrans 测试集上对比音符准确率，目标 BiLSTM-CRF ≥ baseline + 10%。

## 评估配置

| 参数 | 值 |
|------|----|
| 测试集规模 | 769 条（TEST split） |
| GT 音符总数 | 19,955 |
| Onset 容差 | **300ms**（修正后，原因见下） |
| Offset 容差 | 50ms（onset-only 模式，offset 不参与匹配） |

### 评估容差修正说明

HumTrans GT 标注为理想化 MIDI 乐谱（节拍网格对齐），并非与实际演唱对齐的标注，导致 GT onset 与实际发声存在 200–400ms 系统性时间偏移。原始 50ms 容差导致 baseline 虚低（Note Accuracy 仅 2.1%），修正为 300ms 后评估结果更能反映真实性能。

## 方法说明

| 方法 | 特征 | 训练集 | 备注 |
|------|------|--------|------|
| RoundingBaseline | pyin | — | 直接四舍五入量化音高，无学习 |
| BiLSTM-CRF v1 | pyin | 2,000 条 | 第五周训练结果 |
| BiLSTM-CRF v2 | CREPE | 13,080 条 | 替换 pyin 为 CREPE 特征 |
| BiLSTM-CRF v3 | CREPE | 13,080 条 | 新增互相关 GT 对齐（per-sample offset 估算） |
| BiLSTM-CRF v4 | CREPE | 13,080 条 | v3 基础上增至 50 epoch，patience=8 |

## TEST 集评估结果

| 方法 | Note Accuracy | Precision | Recall | F1 | 预测音符数 |
|------|:---:|:---:|:---:|:---:|:---:|
| Baseline (pyin) | 0.308 | 0.154 | 0.308 | 0.199 | 37,021 |
| BiLSTM-CRF v1 (pyin, 2k) | 0.237 | 0.233 | 0.237 | 0.234 | 21,106 |
| BiLSTM-CRF v2 (CREPE, 13k) | 0.272 | 0.288 | 0.272 | 0.279 | 19,111 |
| BiLSTM-CRF v3 (CREPE+对齐) | 0.295 | 0.310 | 0.295 | 0.300 | 18,981 |
| **BiLSTM-CRF v4 (CREPE+对齐+50ep)** | **0.342** | **0.359** | **0.342** | **0.350** | 18,940 |

## 结论

- v4 相比 Baseline：Note Accuracy **+3.4%**（未达 +10% 目标），F1 **+75%**（0.199 → 0.350）
- **Note Accuracy 未超越 Baseline 的根本原因**：Baseline 通过产生大量音符（37k，约为 GT 的 1.85 倍）拉高了 Recall；BiLSTM-CRF 以更精准的预测（19k 音符）换取了远高的 Precision（0.359 vs 0.154），F1 全面领先
- CREPE 替换 pyin（v2）、GT 对齐（v3）、更长训练（v4）均带来稳定提升

## 改进路径分析

| 改进点 | Note Accuracy 提升 |
|--------|-------------------|
| pyin → CREPE 特征 | +0.035 |
| 加入 GT 对齐偏移估算 | +0.023 |
| 训练轮次 30→50，patience 3→8 | +0.047 |

## 交付文件

| 文件 | 说明 |
|------|------|
| `comparison.png` | 5 种方法 Note Accuracy / Precision / F1 对比柱状图 |
| `results.json` | 所有方法完整指标（含 n_pred / n_gt / n_matched） |
