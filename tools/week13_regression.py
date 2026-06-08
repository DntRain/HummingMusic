"""Week 13 regression and stability acceptance runner.

The script executes 31 local checks and writes a machine-readable JSON report
to ../docs/week13_regression_results.json.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT.parent / "docs"
RESULT_PATH = DOCS_DIR / "week13_regression_results.json"
GIT_EXE = shutil.which("git") or r"C:\Users\Administrator\AppData\Git\cmd\git.exe"
PYTHON_EXE = Path(sys.executable)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("FLUID_NO_AUDIO_DRIVERS", "1")


@dataclass
class CaseResult:
    case_id: str
    module: str
    name: str
    input: str
    expected: str
    status: str
    elapsed_ms: float
    output_size_bytes: int
    actual: str
    error: str
    details: dict[str, Any]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run_cmd(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _tail(text: str, limit: int = 1200) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    return str(value)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _make_pitch_data(
    duration: float = 1.0,
    freq: float = 440.0,
    confidence: float = 0.95,
    bpm: float = 120.0,
    step: float = 0.01,
) -> dict[str, Any]:
    import numpy as np

    t = np.arange(0.0, duration, step, dtype=np.float32)
    return {
        "time": t,
        "frequency": np.full(len(t), freq, dtype=np.float32),
        "confidence": np.full(len(t), confidence, dtype=np.float32),
        "bpm": float(bpm),
    }


def _make_test_midi(n_notes: int = 4):
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, name="melody")
    for i in range(n_notes):
        inst.notes.append(
            pretty_midi.Note(
                velocity=92,
                pitch=60 + i,
                start=i * 0.25,
                end=(i + 1) * 0.25,
            )
        )
    midi.instruments.append(inst)
    return midi


def _write_sine_wav(path: Path, duration: float, freq: float = 440.0, amp: float = 0.35) -> None:
    import numpy as np
    import soundfile as sf

    sr = 16000
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    audio = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, audio, sr, subtype="PCM_16")


def _note_count(midi: Any) -> int:
    return sum(len(inst.notes) for inst in midi.instruments)


def _cleanup_path(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    except OSError:
        pass


def _restore_audio_backend(audio_processing: Any, old_backend: Any) -> None:
    if old_backend is None:
        audio_processing._config.get("audio", {}).pop("pitch_backend", None)
    else:
        audio_processing._config.setdefault("audio", {})["pitch_backend"] = old_backend


def case_develop_is_merged() -> dict[str, Any]:
    merged = _run_cmd([GIT_EXE, "merge-base", "--is-ancestor", "origin/develop", "HEAD"], 30)
    head = _run_cmd([GIT_EXE, "rev-parse", "--short", "HEAD"], 30).stdout.strip()
    develop = _run_cmd([GIT_EXE, "rev-parse", "--short", "origin/develop"], 30).stdout.strip()
    _assert(merged.returncode == 0, "origin/develop is not an ancestor of HEAD")
    return {"head": head, "origin_develop": develop}


def case_requirements_streamlit() -> dict[str, Any]:
    text = _read(REPO_ROOT / "requirements.txt").lower()
    _assert("streamlit" in text, "requirements.txt does not include streamlit")
    version = _run_cmd([str(PYTHON_EXE), "-m", "streamlit", "--version"], 30)
    _assert(version.returncode == 0, _tail(version.stderr or version.stdout))
    return {"streamlit_cli": _tail(version.stdout)}


def case_visualizer_exists() -> dict[str, Any]:
    path = REPO_ROOT / "tools" / "visualizer.py"
    _assert(path.exists(), "tools/visualizer.py is missing")
    return {"path": str(path), "bytes": path.stat().st_size}


def case_app_removed() -> dict[str, Any]:
    path = REPO_ROOT / "src" / "app.py"
    _assert(not path.exists(), "src/app.py still exists")
    return {"removed": True}


def case_config_loads() -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load(_read(REPO_ROOT / "config.yaml"))
    _assert(cfg["audio"]["sample_rate"] == 16000, "audio.sample_rate is unexpected")
    _assert("style_transfer" in cfg and "renderer" in cfg, "required config sections missing")
    return {"top_level_sections": sorted(cfg.keys())}


def case_pytest_collect() -> dict[str, Any]:
    result = _run_cmd([str(PYTHON_EXE), "-m", "pytest", "--collect-only", "tests/", "-q"], 180)
    output = result.stdout + result.stderr
    _assert(result.returncode == 0, _tail(output))
    collected = len([line for line in result.stdout.splitlines() if "::" in line])
    match = re.search(r"(\d+)\s+(?:tests?|items?)\s+collected", output)
    if match:
        collected = max(collected, int(match.group(1)))
    _assert(collected >= 30, f"collected only {collected} tests")
    _assert("tests/test_model_bugs.py" in output, "tests/test_model_bugs.py not collected")
    return {"collected": collected}


def case_pytest_full() -> dict[str, Any]:
    result = _run_cmd([str(PYTHON_EXE), "-m", "pytest", "tests/", "-q"], 360)
    output = result.stdout + result.stderr
    _assert(result.returncode == 0, _tail(output))
    match = re.search(r"(\d+)\s+passed", output)
    passed = int(match.group(1)) if match else None
    return {"pytest_passed": passed, "summary": _tail(output, 500)}


def case_model_bugs_included() -> dict[str, Any]:
    path = REPO_ROOT / "tests" / "test_model_bugs.py"
    text = _read(path)
    _assert(path.exists(), "tests/test_model_bugs.py is missing")
    _assert("test_cache_key_stable_for_same_bytes" in text, "Bug-07 cache test missing")
    _assert("test_style_transfer_empty_midi_short_circuit" in text, "Bug-04 style short-circuit test missing")
    return {"path": str(path), "bytes": path.stat().st_size}


def case_extract_pitch_normal() -> dict[str, Any]:
    from src import audio_processing as ap

    with tempfile.TemporaryDirectory(prefix="week13_") as tmp:
        wav = Path(tmp) / "normal.wav"
        _write_sine_wav(wav, 0.8)
        old_backend = ap._config.get("audio", {}).get("pitch_backend")
        ap._config.setdefault("audio", {})["pitch_backend"] = "pyin"
        try:
            result = ap.extract_pitch(str(wav))
        finally:
            _restore_audio_backend(ap, old_backend)
    _assert(len(result["time"]) > 0, "no pitch frames returned")
    _assert(len(result["time"]) == len(result["frequency"]) == len(result["confidence"]), "array length mismatch")
    return {
        "frames": len(result["time"]),
        "valid_frequency_frames": int((~__import__("numpy").isnan(result["frequency"])).sum()),
        "bpm": float(result["bpm"]),
    }


def case_extract_pitch_short() -> dict[str, Any]:
    from src import audio_processing as ap

    with tempfile.TemporaryDirectory(prefix="week13_") as tmp:
        wav = Path(tmp) / "short.wav"
        _write_sine_wav(wav, 0.18)
        old_backend = ap._config.get("audio", {}).get("pitch_backend")
        ap._config.setdefault("audio", {})["pitch_backend"] = "pyin"
        try:
            result = ap.extract_pitch(str(wav))
        finally:
            _restore_audio_backend(ap, old_backend)
    _assert(len(result["time"]) > 0, "short audio returned no frames")
    return {"frames": len(result["time"]), "bpm": float(result["bpm"])}


def case_missing_file_error() -> dict[str, Any]:
    from src.audio_processing import extract_pitch

    missing = REPO_ROOT / "tmp" / f"missing_{uuid.uuid4().hex}.wav"
    try:
        extract_pitch(str(missing))
    except FileNotFoundError as exc:
        return {"error_type": type(exc).__name__}
    raise AssertionError("missing file did not raise FileNotFoundError")


def case_low_energy_error() -> dict[str, Any]:
    import numpy as np
    from src.audio_processing import LowEnergyError, _check_rms_energy

    try:
        _check_rms_energy(np.zeros(16000, dtype=np.float32))
    except LowEnergyError as exc:
        return {"rms_dbfs": exc.rms_dbfs, "threshold_dbfs": exc.threshold_dbfs}
    raise AssertionError("silent input did not raise LowEnergyError")


def case_pitch_backend_available() -> dict[str, Any]:
    from src import audio_processing as ap

    available = {
        "torchcrepe": bool(ap.TORCHCREPE_AVAILABLE),
        "crepe": bool(ap.CREPE_AVAILABLE),
        "pyin": True,
        "configured": ap._config.get("audio", {}).get("pitch_backend", "auto"),
    }
    _assert(any([available["torchcrepe"], available["crepe"], available["pyin"]]), "no F0 backend available")
    return available


def case_quantize_normal() -> dict[str, Any]:
    from src.quantizer import quantize_humming

    midi = quantize_humming(_make_pitch_data(duration=1.0))
    _assert(len(midi.instruments) == 1, "expected a single melody track")
    _assert(_note_count(midi) > 0, "expected at least one note")
    return {"tracks": len(midi.instruments), "notes": _note_count(midi)}


def case_quantize_all_nan() -> dict[str, Any]:
    import numpy as np
    from src.quantizer import quantize_humming

    pitch = _make_pitch_data(duration=1.0)
    pitch["frequency"][:] = np.nan
    midi = quantize_humming(pitch)
    _assert(_note_count(midi) == 0, "all-NaN input should produce empty MIDI")
    return {"notes": _note_count(midi)}


def case_quantize_low_valid_short_circuit() -> dict[str, Any]:
    import numpy as np
    import src.quantizer as q

    pitch = _make_pitch_data(duration=1.0)
    pitch["frequency"][:] = np.nan
    pitch["frequency"][:5] = 440.0
    calls = {"load_model": 0}
    original = q._load_model

    def fake_load_model():
        calls["load_model"] += 1
        return None

    q._load_model = fake_load_model
    try:
        midi = q.quantize_humming(pitch)
    finally:
        q._load_model = original
    _assert(calls["load_model"] == 0, "low-valid input should not load model")
    _assert(_note_count(midi) == 0, "low-valid input should produce empty MIDI")
    return calls


def case_quantize_extreme_pitch_clip() -> dict[str, Any]:
    import numpy as np
    from src.quantizer import quantize_humming

    t = np.arange(0.0, 1.0, 0.01, dtype=np.float32)
    freq = np.where(t < 0.5, 8.18, 12543.85).astype(np.float32)
    pitch = {
        "time": t,
        "frequency": freq,
        "confidence": np.full(len(t), 0.95, dtype=np.float32),
        "bpm": 120.0,
    }
    midi = quantize_humming(pitch)
    pitches = [note.pitch for inst in midi.instruments for note in inst.notes]
    _assert(pitches, "no notes generated for extreme pitches")
    _assert(min(pitches) >= 0 and max(pitches) <= 127, f"pitch out of range: {pitches}")
    return {"min_pitch": min(pitches), "max_pitch": max(pitches), "notes": len(pitches)}


def case_quantize_monotonic_times() -> dict[str, Any]:
    import numpy as np
    from src.quantizer import quantize_humming

    pitch = _make_pitch_data(duration=1.2)
    pitch["frequency"] = np.where(pitch["time"] < 0.6, 440.0, 493.88).astype(np.float32)
    midi = quantize_humming(pitch)
    notes = [note for inst in midi.instruments for note in inst.notes]
    _assert(all(note.start < note.end for note in notes), "note start/end invalid")
    _assert(all(notes[i].start <= notes[i + 1].start for i in range(len(notes) - 1)), "note starts are not monotonic")
    return {"notes": len(notes), "starts": [round(note.start, 3) for note in notes[:5]]}


def case_style(style: str) -> Callable[[], dict[str, Any]]:
    def run() -> dict[str, Any]:
        from src.style_transfer import transfer_style

        midi = _make_test_midi()
        out = transfer_style(midi, style)
        _assert(_note_count(out) >= _note_count(midi), f"{style} output lost notes")
        return {"style": style, "tracks": len(out.instruments), "notes": _note_count(out)}

    return run


def case_style_invalid() -> dict[str, Any]:
    from src.style_transfer import transfer_style

    try:
        transfer_style(_make_test_midi(), "metal")  # type: ignore[arg-type]
    except ValueError as exc:
        return {"error_type": type(exc).__name__}
    raise AssertionError("invalid style did not raise ValueError")


def case_style_empty_short_circuit() -> dict[str, Any]:
    import pretty_midi
    import src.style_transfer as st

    empty = pretty_midi.PrettyMIDI()
    empty.instruments.append(pretty_midi.Instrument(program=0, name="melody"))
    calls = {"vqvae": 0}
    original = st._load_vqvae_model

    def fake_load_model():
        calls["vqvae"] += 1
        return None

    st._load_vqvae_model = fake_load_model
    try:
        out = st.transfer_style(empty, "pop")
    finally:
        st._load_vqvae_model = original
    _assert(calls["vqvae"] == 0, "empty MIDI should skip VQ-VAE")
    _assert(_note_count(out) == 0, "empty MIDI should stay empty")
    return calls


def case_style_fallback_no_mutation() -> dict[str, Any]:
    import src.style_transfer as st

    midi = _make_test_midi()
    original_tracks = len(midi.instruments)
    original_program = midi.instruments[0].program
    out = st._fallback_transfer(midi, "pop")
    _assert(len(midi.instruments) == original_tracks, "fallback mutated input track count")
    _assert(midi.instruments[0].program == original_program, "fallback mutated input program")
    _assert(len(out.instruments) >= original_tracks, "fallback output missing tracks")
    return {"input_tracks": original_tracks, "output_tracks": len(out.instruments)}


def case_style_key_fast() -> dict[str, Any]:
    from src.style_transfer import _estimate_key_fast

    key = _estimate_key_fast(_make_test_midi())
    _assert(0 <= key.tonic.midi <= 11, "key tonic out of range")
    _assert(key.mode in {"major", "minor"}, "key mode invalid")
    return {"tonic": key.tonic.midi, "mode": key.mode}


def case_render_normal() -> dict[str, Any]:
    from src.renderer import render_audio

    wav = Path(render_audio(_make_test_midi()))
    try:
        _assert(wav.exists(), "WAV output missing")
        _assert(wav.stat().st_size > 0, "WAV output is empty")
        return {"path": str(wav), "output_size_bytes": wav.stat().st_size}
    finally:
        _cleanup_path(wav.parent)


def case_render_unique_dirs() -> dict[str, Any]:
    from src.renderer import _ensure_tmp_dir

    d1 = _ensure_tmp_dir()
    d2 = _ensure_tmp_dir()
    try:
        _assert(d1 != d2, "renderer tmp directories are not unique")
        _assert(d1.exists() and d2.exists(), "renderer tmp directories missing")
        return {"dir1": str(d1), "dir2": str(d2)}
    finally:
        _cleanup_path(d1)
        _cleanup_path(d2)


def case_render_empty_midi() -> dict[str, Any]:
    import pretty_midi
    from src.renderer import render_audio

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    midi.instruments.append(pretty_midi.Instrument(program=0, name="melody"))
    wav = Path(render_audio(midi))
    try:
        _assert(wav.exists(), "empty MIDI WAV output missing")
        return {"path": str(wav), "output_size_bytes": wav.stat().st_size}
    finally:
        _cleanup_path(wav.parent)


def case_demo_assets_readable() -> dict[str, Any]:
    demo = REPO_ROOT / "data" / "demo"
    files = ["example.wav", "example.npy", "example.m4a"]
    sizes = {}
    for name in files:
        path = demo / name
        _assert(path.exists(), f"{name} is missing")
        _assert(path.stat().st_size > 0, f"{name} is empty")
        sizes[name] = path.stat().st_size
    return sizes


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def case_streamlit_launch_probe_shutdown() -> dict[str, Any]:
    port = 8501
    _assert(not _port_is_open(port), f"port {port} is already in use before launch")

    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    env.setdefault("FLUID_NO_AUDIO_DRIVERS", "1")
    cmd = [
        str(PYTHON_EXE),
        "-m",
        "streamlit",
        "run",
        str(REPO_ROOT / "tools" / "visualizer.py"),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    status_code = None
    startup_ms = None
    start = time.perf_counter()
    output = ""
    try:
        deadline = time.time() + 45
        while time.time() < deadline:
            if proc.poll() is not None:
                output = proc.communicate(timeout=2)[0]
                raise AssertionError(f"Streamlit exited early: {_tail(output)}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=2) as response:
                    status_code = int(response.status)
                    startup_ms = round((time.perf_counter() - start) * 1000, 2)
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(1.0)
        _assert(status_code == 200, "Streamlit did not return HTTP 200")
    finally:
        proc.terminate()
        try:
            output = proc.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            output = proc.communicate(timeout=5)[0]

    time.sleep(1.0)
    _assert(not _port_is_open(port), f"port {port} is still open after shutdown")
    return {"status_code": status_code, "startup_ms": startup_ms, "process_returncode": proc.returncode, "log_tail": _tail(output, 500)}


CASES: list[tuple[str, str, str, str, str, Callable[[], dict[str, Any]]]] = [
    ("R01", "baseline", "develop commit merged", "local bridge HEAD", "origin/develop is included", case_develop_is_merged),
    ("R02", "baseline", "requirements use Streamlit", "requirements.txt + streamlit CLI", "streamlit dependency is installed", case_requirements_streamlit),
    ("R03", "baseline", "visualizer exists", "tools/visualizer.py", "Streamlit entrypoint exists", case_visualizer_exists),
    ("R04", "baseline", "legacy Gradio app removed", "src/app.py", "legacy entrypoint is absent", case_app_removed),
    ("R05", "baseline", "config loads", "config.yaml", "YAML loads required sections", case_config_loads),
    ("R06", "unit", "pytest collect-only", "pytest --collect-only tests/ -q", ">=30 cases collected and model bug tests included", case_pytest_collect),
    ("R07", "unit", "full pytest", "pytest tests/ -q", "all unit tests pass", case_pytest_full),
    ("R08", "unit", "model bug regression tests present", "tests/test_model_bugs.py", "Bug-04/Bug-07 tests exist", case_model_bugs_included),
    ("R09", "audio", "normal audio extracts pitch", "0.8s sine WAV", "pitch arrays returned", case_extract_pitch_normal),
    ("R10", "audio", "short audio does not crash", "0.18s sine WAV", "pitch arrays returned", case_extract_pitch_short),
    ("R11", "audio", "missing file error", "non-existent WAV path", "FileNotFoundError", case_missing_file_error),
    ("R12", "audio", "low energy gate", "zero audio array", "LowEnergyError", case_low_energy_error),
    ("R13", "audio", "F0 backend availability", "auto backend flags", "at least one backend available", case_pitch_backend_available),
    ("R14", "quantizer", "normal pitch to MIDI", "1s A4 pitch_data", "single melody track with notes", case_quantize_normal),
    ("R15", "quantizer", "all NaN input", "all-NaN frequency", "empty MIDI", case_quantize_all_nan),
    ("R16", "quantizer", "low-valid short-circuit", "5 percent valid frequency", "model loader is skipped", case_quantize_low_valid_short_circuit),
    ("R17", "quantizer", "extreme pitch clipping", "MIDI 0/127 edge frequencies", "pitches stay in [0,127]", case_quantize_extreme_pitch_clip),
    ("R18", "quantizer", "monotonic note times", "two-note pitch_data", "note times increase", case_quantize_monotonic_times),
    ("R19", "style", "pop style callable", "single-track MIDI", "returns PrettyMIDI-like output", case_style("pop")),
    ("R20", "style", "jazz style callable", "single-track MIDI", "returns PrettyMIDI-like output", case_style("jazz")),
    ("R21", "style", "classical style callable", "single-track MIDI", "returns PrettyMIDI-like output", case_style("classical")),
    ("R22", "style", "folk style callable", "single-track MIDI", "returns PrettyMIDI-like output", case_style("folk")),
    ("R23", "style", "invalid style rejected", "style='metal'", "ValueError", case_style_invalid),
    ("R24", "style", "empty MIDI short-circuit", "empty PrettyMIDI", "skips VQ-VAE and stays empty", case_style_empty_short_circuit),
    ("R25", "style", "fallback does not mutate input", "fallback transfer", "input track count/program unchanged", case_style_fallback_no_mutation),
    ("R26", "style", "fast key estimate", "test melody", "tonic and mode are valid", case_style_key_fast),
    ("R27", "renderer", "normal MIDI renders WAV", "4-note MIDI", "WAV file exists and is non-empty", case_render_normal),
    ("R28", "renderer", "unique output directories", "two _ensure_tmp_dir calls", "directories differ", case_render_unique_dirs),
    ("R29", "renderer", "empty MIDI render stability", "empty melody track", "no crash and output path exists", case_render_empty_midi),
    ("R30", "frontend", "demo assets readable", "data/demo example files", "wav/npy/m4a are readable", case_demo_assets_readable),
    ("R31", "frontend", "Streamlit launch/probe/shutdown", "port 8501", "HTTP 200 then clean shutdown", case_streamlit_launch_probe_shutdown),
]


def run_case(case: tuple[str, str, str, str, str, Callable[[], dict[str, Any]]]) -> CaseResult:
    case_id, module, name, input_desc, expected, func = case
    start = time.perf_counter()
    details: dict[str, Any] = {}
    output_size = 0
    actual = ""
    error = ""
    status = "PASS"
    try:
        details = func() or {}
        output_size = int(details.pop("output_size_bytes", 0) or 0)
        actual = "passed"
    except Exception as exc:  # noqa: BLE001 - regression runner must isolate failures.
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
        actual = "failed"
        details = {"traceback": traceback.format_exc(limit=4)}
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    return CaseResult(
        case_id=case_id,
        module=module,
        name=name,
        input=input_desc,
        expected=expected,
        status=status,
        elapsed_ms=elapsed_ms,
        output_size_bytes=output_size,
        actual=actual,
        error=error,
        details=details,
    )


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    results = [run_case(case) for case in CASES]
    finished = time.strftime("%Y-%m-%d %H:%M:%S")
    passed = sum(1 for result in results if result.status == "PASS")
    failed = len(results) - passed
    elapsed_ms = round(sum(result.elapsed_ms for result in results), 2)
    payload = {
        "task": "week13_regression_acceptance",
        "repo": str(REPO_ROOT),
        "started_at": started,
        "finished_at": finished,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / max(len(results), 1), 4),
            "elapsed_ms_sum": elapsed_ms,
            "p0_p1_blockers": failed,
        },
        "results": [_json_safe(asdict(result)) for result in results],
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {RESULT_PATH}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
