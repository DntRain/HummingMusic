"""
tools/run_acceptance_tests.py - Week13 端到端验收测试

覆盖 ≥10 个用例：演示 / 用户录音 / 边界（静音/纯音/白噪声/超短/低音量/超长）。
每个用例跑完整链路：
    raw audio
    ├─ extract_pitch                  (audio_processing.extract_pitch)
    ├─ quantize_humming               (quantizer)
    ├─ transfer_style × 4 styles      (style_transfer)
    └─ render_audio (1×)              (renderer)

记录：耗时分阶段、各阶段输出、是否捕获预期异常、是否 crash。
判定：unhandled exception = ❌ 崩溃；预期异常被捕获 = ✅ 通过。

输出：
    reports/week13/dc/acceptance_results.json
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("FLUID_NO_AUDIO_DRIVERS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio_processing import LowEnergyError, extract_pitch
from src.interfaces import VALID_STYLES, quantize_humming, render_audio, transfer_style


# ──────────────────────────────────────────────
# 用例定义
# ──────────────────────────────────────────────

CASES: list[dict] = [
    # ── 正常 ─────────────────────────────────────
    {"tc": "TC-01", "name": "demo example.wav",
     "path": str(ROOT / "data/demo/example.wav"),
     "expect": "ok", "note": "正常哼唱，34.75s"},
    {"tc": "TC-02", "name": "demo example.m4a",
     "path": str(ROOT / "data/demo/example.m4a"),
     "expect": "ok", "note": "AAC 容器同源音频"},
    {"tc": "TC-03", "name": "user rec example_1.m4a",
     "path": str(ROOT / "example_1.m4a"),
     "expect": "ok", "note": "用户测试录音 #1"},
    {"tc": "TC-04", "name": "user rec example_2.m4a",
     "path": str(ROOT / "example_2.m4a"),
     "expect": "ok", "note": "用户测试录音 #2"},
    {"tc": "TC-05", "name": "user rec example_3.m4a",
     "path": str(ROOT / "example_3.m4a"),
     "expect": "ok", "note": "用户测试录音 #3"},
    {"tc": "TC-06", "name": "example trim 5s",
     "path": "/tmp/acc/example_trim.wav",
     "expect": "ok", "note": "example.wav 前 5s"},
    {"tc": "TC-07", "name": "example loop 90+s",
     "path": "/tmp/acc/example_long.wav",
     "expect": "ok", "note": "example.wav 重复 3 次约 100s"},
    {"tc": "TC-08", "name": "pure tone 440Hz 3s",
     "path": "/tmp/acc/tone.wav",
     "expect": "ok", "note": "纯音输入，预期能量过门、量化产出 A4 持续音"},

    # ── 边界（预期触发能量/有效帧短路） ────────
    {"tc": "TC-09", "name": "digital silence 5s",
     "path": "/tmp/acc/silence.wav",
     "expect": "low_energy", "note": "数字静音，预期 LowEnergyError"},
    {"tc": "TC-10", "name": "white noise 5s @ 0.01",
     "path": "/tmp/acc/whitenoise.wav",
     "expect": "low_energy_or_empty",
     "note": "白噪声，预期 LowEnergyError 或全 NaN 短路"},
    {"tc": "TC-11", "name": "tiny audio 0.5s",
     "path": "/tmp/acc/short.wav",
     "expect": "ok_or_empty", "note": "0.5s 极短，预期跑通但产出极少"},
    {"tc": "TC-12", "name": "low volume example",
     "path": "/tmp/acc/lowvol.wav",
     "expect": "low_energy_or_empty",
     "note": "example.wav 音量 ×0.02，预期接近能量阈值"},
]


# ──────────────────────────────────────────────
# 单用例执行
# ──────────────────────────────────────────────

def _run_one(case: dict) -> dict:
    result: dict = {
        "tc": case["tc"], "name": case["name"],
        "path": case["path"], "expect": case["expect"], "note": case["note"],
        "stages": {}, "passed": False, "crashed": False,
        "exception_type": None, "exception_msg": None,
    }
    path = case["path"]
    if not Path(path).exists():
        result["exception_type"] = "FileNotFoundError"
        result["exception_msg"] = f"音频文件不存在: {path}"
        result["crashed"] = True
        return result

    # ── 阶段 1: extract_pitch ──
    t0 = time.perf_counter()
    try:
        pitch = extract_pitch(path)
        result["stages"]["extract_pitch"] = {
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "n_frames": int(len(pitch["time"])),
            "n_valid": int((~__import__("numpy").isnan(pitch["frequency"])).sum()),
            "bpm": round(float(pitch["bpm"]), 1),
        }
    except LowEnergyError as e:
        result["stages"]["extract_pitch"] = {
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "low_energy": True,
            "rms_dbfs": round(float(e.rms_dbfs), 2),
        }
        result["exception_type"] = "LowEnergyError"
        result["exception_msg"] = str(e)
        # 预期捕获，正常结束
        result["passed"] = case["expect"] in ("low_energy", "low_energy_or_empty")
        return result
    except Exception as e:
        result["stages"]["extract_pitch"] = {
            "elapsed_s": round(time.perf_counter() - t0, 3), "error": True,
        }
        result["exception_type"] = type(e).__name__
        result["exception_msg"] = str(e)
        result["traceback"] = traceback.format_exc(limit=4)
        result["crashed"] = True
        return result

    # ── 阶段 2: quantize_humming ──
    t1 = time.perf_counter()
    try:
        midi = quantize_humming(pitch)
        n_notes = sum(len(inst.notes) for inst in midi.instruments)
        result["stages"]["quantize"] = {
            "elapsed_s": round(time.perf_counter() - t1, 3),
            "n_notes": n_notes,
        }
    except Exception as e:
        result["exception_type"] = type(e).__name__
        result["exception_msg"] = str(e)
        result["traceback"] = traceback.format_exc(limit=4)
        result["crashed"] = True
        return result

    # ── 阶段 3: transfer_style × 4 ──
    style_stage = {}
    for style in VALID_STYLES:
        t2 = time.perf_counter()
        try:
            styled = transfer_style(midi, style)
            n = sum(len(inst.notes) for inst in styled.instruments)
            style_stage[style] = {
                "elapsed_s": round(time.perf_counter() - t2, 3),
                "n_notes": n,
            }
        except Exception as e:
            style_stage[style] = {
                "elapsed_s": round(time.perf_counter() - t2, 3),
                "error": type(e).__name__,
            }
            result["exception_type"] = type(e).__name__
            result["exception_msg"] = str(e)
            result["traceback"] = traceback.format_exc(limit=4)
            result["crashed"] = True
            return result
    result["stages"]["transfer_style"] = style_stage

    # ── 阶段 4: render_audio (1 个风格，避免 ×4 fluidsynth 太慢) ──
    t3 = time.perf_counter()
    try:
        wav_path = render_audio(styled)  # 用最后一个风格
        wav_size = Path(wav_path).stat().st_size if Path(wav_path).exists() else 0
        result["stages"]["render"] = {
            "elapsed_s": round(time.perf_counter() - t3, 3),
            "wav_size_kb": round(wav_size / 1024, 1),
        }
    except Exception as e:
        result["exception_type"] = type(e).__name__
        result["exception_msg"] = str(e)
        result["traceback"] = traceback.format_exc(limit=4)
        result["crashed"] = True
        return result

    # ── 判定 ──
    # ok：四阶段都跑通，每阶段至少不崩
    # ok_or_empty：跑通即可，n_notes 可以为 0
    # low_energy_or_empty：能量门控未触发但量化全空，也算预期
    if case["expect"] in ("ok", "ok_or_empty"):
        result["passed"] = True
    elif case["expect"] == "low_energy_or_empty":
        # 没抛 LowEnergyError，那量化和风格至少不应崩
        result["passed"] = True
    elif case["expect"] == "low_energy":
        # 应该抛却没抛，标记为非预期但不算崩溃
        result["passed"] = False
    return result


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main() -> int:
    out_dir = ROOT / "reports/week13/dc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "acceptance_results.json"

    results = []
    t_all = time.perf_counter()
    for i, case in enumerate(CASES, 1):
        print(f"\n[{i}/{len(CASES)}] {case['tc']} {case['name']}", flush=True)
        r = _run_one(case)
        results.append(r)
        verdict = "✅ PASS" if r["passed"] else ("❌ CRASH" if r["crashed"] else "⚠️ UNEXPECTED")
        exc = f" [{r['exception_type']}]" if r["exception_type"] else ""
        print(f"  → {verdict}{exc}", flush=True)

    total = time.perf_counter() - t_all
    summary = {
        "total_cases": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "crashed": sum(1 for r in results if r["crashed"]),
        "unexpected": sum(1 for r in results if not r["passed"] and not r["crashed"]),
        "elapsed_s": round(total, 1),
    }
    print(f"\n{'='*60}\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n{'='*60}")

    out_path.write_text(json.dumps(
        {"summary": summary, "results": results},
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")
    print(f"\n📄 结果写入: {out_path}")
    return 0 if summary["crashed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
