"""单次启动一个 UI（before/after 模式）用于截图。"""
import argparse
import importlib.util
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--module_path", required=True,
                help="app.py 的绝对路径 (before 用 /tmp/app_before.py，after 用 src/app.py)")
ap.add_argument("--port", type=int, required=True)
args = ap.parse_args()

# 让 from src.interfaces ... 等 import 能解析
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

spec = importlib.util.spec_from_file_location("app_under_test", args.module_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

app = mod.create_ui()
app.launch(server_name="127.0.0.1", server_port=args.port,
           share=False, quiet=True, inbrowser=False)
