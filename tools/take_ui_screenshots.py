"""启动 before/after 两版 UI，用 playwright 抓首屏截图。"""
import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent


def wait_port(port: int, timeout: float = 60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(1.0)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.5)
    raise TimeoutError(f"端口 {port} 未在 {timeout}s 内响应")


def start_app(module_path: str, port: int) -> subprocess.Popen:
    log = open(ROOT / f"reports/week12/ui_screenshots/_app_{port}.log", "wb")
    return subprocess.Popen(
        [sys.executable, "-m", "tools._run_one_ui",
         "--module_path", module_path, "--port", str(port)],
        cwd=str(ROOT), stdout=log, stderr=log,
    )


def shot(page, out: Path, *, full=True):
    out.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(out), full_page=full)
    print(f"  saved: {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="/tmp/app_before.py")
    ap.add_argument("--after", default=str(ROOT / "src/app.py"))
    ap.add_argument("--out_dir", default=str(ROOT / "reports/week12/ui_screenshots"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    before_port, after_port = 7861, 7862
    procs = []
    try:
        print("启动 BEFORE UI ...")
        procs.append(start_app(args.before, before_port))
        wait_port(before_port)
        print("启动 AFTER UI ...")
        procs.append(start_app(args.after, after_port))
        wait_port(after_port)
        time.sleep(3)  # 让首次 Hugging Face 字体/资源加载完

        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(viewport={"width": 1440, "height": 900},
                                      device_scale_factor=1.5)
            page = ctx.new_page()

            print("截图 BEFORE 首屏...")
            page.goto(f"http://127.0.0.1:{before_port}", wait_until="networkidle")
            page.wait_for_timeout(2000)
            shot(page, out_dir / "00_before_home.png")

            print("截图 AFTER 首屏...")
            page.goto(f"http://127.0.0.1:{after_port}", wait_until="networkidle")
            page.wait_for_timeout(2000)
            shot(page, out_dir / "01_after_home.png")

            print("截图 AFTER 风格下拉打开...")
            try:
                page.get_by_label("🎨 目标风格").click(timeout=5000)
                page.wait_for_timeout(800)
                shot(page, out_dir / "02_after_style_dropdown.png", full=False)
                page.keyboard.press("Escape")
            except Exception as e:
                print(f"  下拉截图失败（非阻断）: {e}")

            print("截图 AFTER 输入变更后的过期提示...")
            try:
                page.locator("input.dropdown-input, input[role='combobox']").first.click(timeout=5000)
                page.wait_for_timeout(500)
                page.get_by_text("爵士", exact=False).first.click(timeout=5000)
                page.wait_for_timeout(800)
                shot(page, out_dir / "03_after_stale_warning.png")
            except Exception as e:
                print(f"  过期提示截图跳过: {e}")

            browser.close()
            print("完成")
    finally:
        for p_ in procs:
            p_.terminate()
            try:
                p_.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p_.kill()


if __name__ == "__main__":
    main()
