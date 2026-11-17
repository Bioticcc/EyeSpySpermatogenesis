"""Make the pictures for the annotation guide (the landing page at ``/``).

Usage::

    PYTHON_WITH_PLAYWRIGHT scripts/make_guide_screenshots.py WORKSPACE

Writes ``WORKSPACE/guide/*.png``. The pictures show tissue, so they stay in the
Git-ignored workspace (never in the repository); ``scripts/sync.sh push-files``
uploads them to the hosted page.

The page runs on a throwaway copy of ``labels.sqlite`` in a temporary folder, so
the example login and example label never reach the real database. Needs
Playwright with Chromium, which is not a project dependency; install it in a
separate environment, e.g.::

    python3 -m venv /tmp/pw && /tmp/pw/bin/pip install playwright==1.48.0
    /tmp/pw/bin/playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
NAME = "Example annotator"
# Sections chosen for the pictures: one partly staged (map), one with someone
# else's on-screen labels (Annotate). Falls back to the first section.
MAP_SECTION = "#106_WR2_44d_7-1"
ANNOTATE_SECTION = "#105_WR2_44d_6-1"
HIGHLIGHT = (
    ".g-hl { outline: 3px solid #a15d14 !important; outline-offset: 4px;"
    " border-radius: 8px; }"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def copy_workspace(ws: Path, tmp: Path) -> Path:
    """Database copied (backup API, consistent while the page runs); images linked."""
    out = tmp / "ws"
    out.mkdir()
    src = sqlite3.connect(f"file:{ws / 'labels.sqlite'}?mode=ro", uri=True)
    dst = sqlite3.connect(out / "labels.sqlite")
    src.backup(dst)
    src.close()
    dst.close()
    for name in ("images", "paper", "review"):
        if (ws / name).exists():
            (out / name).symlink_to((ws / name).resolve())
    if (ws / "workspace.json").exists():
        shutil.copy2(ws / "workspace.json", out / "workspace.json")
    return out


def start_server(ws: Path, port: int) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    proc = subprocess.Popen(
        [sys.executable, "-m", "eyespy_spermatogenesis.staging.annotate.server",
         str(ws), "--port", str(port), "--no-browser"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/config", timeout=1)
            return proc
        except OSError:
            time.sleep(0.2)
    proc.terminate()
    raise SystemExit("The page did not start")


def pick(page, select_id: str, label: str):
    """Choose the option whose text is ``label`` (or the second option)."""
    options = page.locator(f"#{select_id} option").all_text_contents()
    text = label if label in options else options[min(1, len(options) - 1)]
    page.select_option(f"#{select_id}", label=text)


def mark(page, *selectors):
    page.evaluate(
        "sels => { for (const e of document.querySelectorAll('.g-hl'))"
        " e.classList.remove('g-hl');"
        " for (const s of sels) document.querySelector(s)?.classList.add('g-hl'); }",
        list(selectors),
    )


def wait_images(page):
    page.wait_for_function(
        "[...document.images].filter(i => i.offsetParent).every(i => i.complete)"
    )
    page.wait_for_timeout(400)


def login(page, base, mode):
    if base:
        page.goto(base + "/app")
    page.wait_for_selector("#view-login:not([hidden])")
    page.fill("#name", NAME)
    page.check(f"input[name=mode][value={mode}]")


def shoot(ws: Path, base: str, out: Path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 860})
        page.add_init_script(
            "document.addEventListener('DOMContentLoaded', () => {"
            " const s = document.createElement('style');"
            f" s.textContent = {json.dumps(HIGHLIGHT)};"
            " document.head.append(s); });"
        )

        # 1. Sign in
        login(page, base, "staging")
        mark(page, "#name", ".mode-card:has(input[value=staging])", "#start")
        page.screenshot(path=out / "01-login.png")

        # 2. Progress
        page.click("#start")
        page.wait_for_selector("#view-progress:not([hidden]) #by-source tr.clickable")
        mark(page, "#metrics", "#by-source")
        page.screenshot(path=out / "02-progress.png")

        # 4. Section map (taken before 3 so the list filters end on Annotate)
        mark(page)
        row = page.locator("#by-source tr.clickable", has_text=MAP_SECTION)
        (row if row.count() else page.locator("#by-source tr.clickable")).first.click()
        page.wait_for_function(
            "!document.getElementById('sm').hidden &&"
            " !document.getElementById('sm-summary').textContent.startsWith('Loading')"
        )
        page.wait_for_timeout(800)
        page.screenshot(path=out / "04-section-map.png")
        page.keyboard.press("Escape")

        # 3. Choose from a list
        page.click(".nav-item[data-view=annotate]")
        pick(page, "f-source", MAP_SECTION)
        page.select_option("#f-queue", "mine")
        wait_images(page)
        mark(page, "#f-source", "#f-queue")
        page.screenshot(path=out / "03-choose-list.png")

        # 5. Stage a tubule: one with someone else's label, then give an example label
        pick(page, "f-source", ANNOTATE_SECTION)
        page.select_option("#f-queue", "all")
        wait_images(page)
        page.keyboard.press("6")
        page.wait_for_timeout(500)
        page.keyboard.press("ArrowLeft")
        wait_images(page)
        mark(page, "#keys", ".mine", "#others")
        page.screenshot(path=out / "05-annotate.png")

        # 6. Saved
        page.wait_for_function(
            "document.getElementById('save-state').textContent.includes('All saved')"
        )
        mark(page)
        page.locator(".sidebar-foot").screenshot(path=out / "06-saved.png")

        # 7. Model review overview (EyeSpy tab: the selected strategy)
        page.click("#logout")  # returns to the sign-in screen; no reload
        login(page, None, "model_review")
        page.click("#start")
        page.wait_for_selector("#view-overview:not([hidden])")
        page.click("#ov-tab-eyespy")
        page.wait_for_selector("#es-grid img")
        wait_images(page)
        page.evaluate("document.getElementById('toast').hidden = true")
        page.screenshot(path=out / "07-overview.png")
        browser.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("workspace", type=Path)
    a = p.parse_args(argv)
    ws = a.workspace.resolve()
    out = ws / "guide"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eyespy-guide-") as tmp:
        copy = copy_workspace(ws, Path(tmp))
        port = free_port()
        proc = start_server(copy, port)
        try:
            shoot(copy, f"http://127.0.0.1:{port}", out)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    print(f"Guide pictures: {out}")


if __name__ == "__main__":
    main()
