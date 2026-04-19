"""
train/extract_features.py - 离线特征预提取脚本

批量提取 HumTrans 数据集的帧级特征，保存为 .npy 文件。
优先使用 CREPE（精度高），不可用时回退到 librosa.pyin。

运行示例：
    python -m train.extract_features \\
        --data_root /run/media/DontRain/DATA_NANO/HumTrans \\
        --feat_dir data/features \\
        --splits TRAIN VALID TEST \\
        --workers 4
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

FRAME_STEP_MS = 10


def extract_one(
    key: str,
    wav_dir: Path,
    feat_dir: Path,
    force: bool = False,
) -> bool:
    """
    提取单条样本的特征并保存为 .npy。

    Args:
        key: 样本 ID。
        wav_dir: WAV 目录。
        feat_dir: 特征输出目录。
        force: 已存在时是否重新提取。

    Returns:
        bool: 成功返回 True，跳过/失败返回 False。
    """
    import librosa

    out_path = feat_dir / f"{key}.npy"
    if out_path.exists() and not force:
        return False  # 已存在，跳过

    wav_path = wav_dir / f"{key}.wav"
    if not wav_path.exists():
        logger.warning("WAV 不存在: %s", wav_path)
        return False

    try:
        audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)

        try:
            import crepe
            time, freq, conf, _ = crepe.predict(
                audio, sr,
                step_size=FRAME_STEP_MS,
                viterbi=True,
                model_capacity="full",
            )
        except ImportError:
            # 回退：librosa pYIN（无需 tensorflow）
            hop = int(sr * FRAME_STEP_MS / 1000.0)
            freq, _, voiced_prob = librosa.pyin(
                audio, sr=sr,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C6"),
                hop_length=hop,
                fill_na=0.0,
            )
            conf = voiced_prob.astype(np.float64)
            time = np.arange(len(freq)) * (FRAME_STEP_MS / 1000.0)

        # F0 → MIDI 编号
        with np.errstate(divide="ignore", invalid="ignore"):
            midi_notes = np.where(
                freq > 0,
                69.0 + 12.0 * np.log2(freq / 440.0),
                0.0,
            ).astype(np.float32)

        valid = (freq > 0).astype(np.float32)

        # 节拍位置
        bpm, _ = librosa.beat.beat_track(y=audio, sr=sr)
        bpm = float(np.atleast_1d(bpm)[0])
        beat_dur = 60.0 / max(bpm, 1.0)
        beat_pos = ((time % beat_dur) / beat_dur).astype(np.float32)

        features = np.stack(
            [midi_notes, valid, conf.astype(np.float32), beat_pos],
            axis=-1,
        )  # (T, 4)

        np.save(str(out_path), features)
        return True

    except Exception as e:
        logger.error("提取失败 %s: %s", key, e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="HumTrans 特征预提取")
    parser.add_argument(
        "--data_root",
        default="/run/media/DontRain/DATA_NANO/HumTrans",
        help="HumTrans 数据集根目录",
    )
    parser.add_argument(
        "--feat_dir",
        default="data/features",
        help="特征输出目录",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["TRAIN", "VALID", "TEST"],
        help="要提取的数据集分割",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行进程数",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新提取（覆盖已有文件）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个 split 最多提取的样本数（调试用）",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    feat_dir = Path(args.feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)

    split_json = data_root / "train_valid_test_keys.json"
    wav_dir = data_root / "all_wav" / "wav_data_sync_with_midi"

    with open(split_json, "r", encoding="utf-8") as f:
        all_splits = json.load(f)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    for split in args.splits:
        keys = all_splits[split.upper()]
        if args.max_samples:
            keys = keys[:args.max_samples]

        logger.info("开始提取 [%s]: %d 条", split, len(keys))
        done = skipped = failed = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(extract_one, key, wav_dir, feat_dir, args.force): key
                for key in keys
            }
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    done += 1
                else:
                    # 区分跳过和失败（失败时 result 也是 False，日志已打印）
                    out = feat_dir / f"{futures[future]}.npy"
                    if out.exists():
                        skipped += 1
                    else:
                        failed += 1

                if (i + 1) % 100 == 0:
                    logger.info(
                        "[%s] 进度: %d/%d (新增=%d 跳过=%d 失败=%d)",
                        split, i + 1, len(keys), done, skipped, failed,
                    )

        logger.info(
            "[%s] 完成: 新增=%d 跳过=%d 失败=%d",
            split, done, skipped, failed,
        )


if __name__ == "__main__":
    main()
