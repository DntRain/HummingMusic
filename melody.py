import numpy as np
from scipy.io import wavfile


def generate_complex_audio(filename="complex_melody.wav"):
    fs = 44100  # 采样率
    # 定义音符频率 (C4 到 A4)
    notes = {
        'C': 261.63, 'D': 293.66, 'E': 329.63,
        'F': 349.23, 'G': 392.00, 'A': 440.00
    }

    # 编排一段旋律：小星星前两句 (C C G G A A G)
    melody_sequence = ['C', 'C', 'G', 'G', 'A', 'A', 'G']
    note_duration = 0.5  # 每个音符 0.5 秒

    full_audio = []

    for note in melody_sequence:
        freq = notes[note]
        t = np.linspace(0, note_duration, int(fs * note_duration), endpoint=False)

        # 1. 基础波形（正弦波）
        wave = np.sin(2 * np.pi * freq * t)

        # 2. 加入泛音 (让声音更厚实，像乐器)
        wave += 0.5 * np.sin(2 * np.pi * (2 * freq) * t)  # 高一个八度
        wave += 0.2 * np.sin(2 * np.pi * (3 * freq) * t)  # 再高一点

        # 3. 加入“包络线” (让声音有弹奏感，开头响，结尾慢慢消失)
        envelope = np.exp(-3 * t / note_duration)
        wave = wave * envelope

        full_audio.append(wave)

    # 合并所有音符
    final_wave = np.concatenate(full_audio)

    # 归一化并转换为 16bit PCM
    final_wave = final_wave / np.max(np.abs(final_wave))
    final_wave = (final_wave * 32767).astype(np.int16)

    wavfile.write(filename, fs, final_wave)
    print(f"✅ 复杂旋律已生成：{filename}")


if __name__ == "__main__":
    generate_complex_audio()