import fluidsynth
import os
import wave
import numpy as np
import time

# 路径锁定：刚才你搬运到 D:\shit 下的文件
SF2_PATH = r'D:\shit\PCLite.sf2'
# 输出到你的项目目录
OUT_WAV = r'D:\shit\HummingMusic\W6_Final_Voice.wav'


def make_it_sound():
    print("🎹 正在加载音色库 PCLite.sf2...")

    if not os.path.exists(SF2_PATH):
        print(f"❌ 还是没找到文件！请确认你已经把 PCLite.sf2 复制到了 D:\shit\ 文件夹下。")
        return

    # 初始化渲染引擎
    fs = fluidsynth.Synth()
    # 强制不找声卡，只管写文件
    fs.start(driver="file")

    sfid = fs.sfload(SF2_PATH)
    if sfid == -1:
        print("❌ 错误：音色库加载失败。请检查文件是否完整（大小应大于 2MB）。")
        return

    fs.program_select(0, sfid, 0, 0)

    print("🎵 正在弹奏 C Major 和弦并写入 WAV...")
    # 弹三个音：C, E, G
    fs.noteon(0, 60, 100)
    fs.noteon(0, 64, 100)
    fs.noteon(0, 67, 100)

    # 渲染 3 秒
    sr = 44100
    samples = fs.get_samples(sr * 3)
    audio_data = np.array(samples).astype(np.int16)

    # 确保文件夹存在
    os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)

    with wave.open(OUT_WAV, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_data.tobytes())

    print("\n" + "=" * 40)
    print("✨ 第 6 周任务圆满收官！")
    print(f"✅ 音频已生成：{OUT_WAV}")
    print("💡 现在去文件夹里双击这个 WAV，你应该能听到钢琴声了！")
    print("=" * 40)


if __name__ == "__main__":
    make_it_sound()