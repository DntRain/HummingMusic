# 第五周：BiLSTM-CRF v1 训练

## 任务目标

使用 HumTrans 2000 条标注数据训练 BiLSTM-CRF v1，验证集音符准确率目标 ≥ baseline + 5%。

## 模型配置

| 参数 | 值 |
|------|----|
| 训练集规模 | 2000 条（pyin 特征） |
| 模型结构 | BiLSTM-CRF，hidden=128，2层，dropout=0.3 |
| Epochs | 30 |
| 学习率 | 1e-3（ReduceLROnPlateau，patience=3，factor=0.5） |
| Batch size | 32 |
| 特征 | pyin 音高 + voiced 概率 + 置信度 + MIDI 音高（4维） |
| 标注方案 | BIO 序列标注 |
| 损失函数 | CRF loss × 0.2 + 加权 CE loss（B×50, I×0.5, O×1） |

## 关键工程改动

- **类别不平衡处理**：B 类（音符起始）帧占比仅约 1.2%，通过辅助 CE 损失加权（B×50）缓解
- **评估方式**：改用 B 类发射分数局部峰值检测（`scipy.signal.find_peaks`），替代直接 argmax 解码
- **Checkpoint 格式**：包含 `epoch`、`optimizer_state_dict`、`best_acc` 元信息

## 评估结果

> 评估使用 onset 容差 50ms（当时的评估配置）

| 方法 | Note Accuracy |
|------|--------------|
| RoundingBaseline (pyin) | 2.1% |
| BiLSTM-CRF v1 (pyin, 2k) | **9.54%** |
| Δ | **+7.4%** ✓ |

目标为 ≥ +5%，**达标**。

> **注**：第六周对比实验中发现评估存在系统性偏差（GT 与实际演唱时间偏移 200–400ms，50ms 容差过严），将容差修正至 300ms 后，整体数值显著提升，详见 week6。

## 交付文件

| 文件 | 说明 |
|------|------|
| `bilstm_crf.pt` | BiLSTM-CRF v1 最优检查点（最优 val_acc=9.54%） |
| `training_curve_v1.png` | 训练过程 loss 与 val_acc 曲线 |
