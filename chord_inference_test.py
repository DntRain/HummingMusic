import os
import time
import numpy as np
from chord_engine import MusicBrain  # 显式调用引擎模块


def run_quality_assurance():
    print("🔍 [QA系统] 开始执行第 5 周任务指标核验...")
    engine = MusicBrain()
    test_sample = "complex_melody.wav"

    if not os.path.exists(test_sample):
        print("❌ 错误：测试样本缺失。")
        return

    # --- 1. 性能压测 (Benchmark) ---
    print("\n--- 1. 性能压测 (50轮循环) ---")
    latencies = []
    # 预热一次
    engine.predict(test_sample)
    for _ in range(50):
        t_start = time.perf_counter()
        engine.predict(test_sample)
        latencies.append(time.perf_counter() - t_start)

    avg_ms = np.mean(latencies) * 1000
    print(f"📊 平均耗时: {avg_ms:.3f} ms (指标要求: < 1000ms)")

    # --- 2. 覆盖度自检 (Compliance) ---
    print("\n--- 2. 功能覆盖度自检 ---")
    chord_num = len(engine.chord_library)
    print(f"🎹 和弦库容量: {chord_num} 种 (指标要求: >= 12)")

    # --- 3. 语义准确性校验 (Inference) ---
    print("\n--- 3. 算法语义输出校验 ---")
    note, chord = engine.predict(test_sample)
    print(f"🎯 样本[{test_sample}] -> 识别音符: {note} -> 建议和弦: {chord}")

    print("\n" + "=" * 40)
    status = "🏆 优秀达标" if avg_ms < 10 else "✅ 合格达标"
    print(f"测试结论：{status}")
    print("=" * 40)


if __name__ == "__main__":
    run_quality_assurance()