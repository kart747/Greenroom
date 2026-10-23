"""Start Greenroom: one command, one window.

    python greenroom.py              # opens the app in its own desktop window
    python greenroom.py --browser    # use the default web browser instead
    python greenroom.py --port 9000

What happens:
  1. config is loaded (hardware defaults <- config.json), see coach/config.py
  2. the FastAPI server starts on 127.0.0.1 in a background thread; models
     load in the background so the window appears right away
  3. a desktop window (pywebview) shows the app. If pywebview can't run here,
     the default browser opens the same address instead.
Closing the window stops the server.
"""

from __future__ import annotations

import os

# Runtime is offline: tell the Hugging Face libraries never to look online
# (the models were fetched once by setup_models.py and load from models/).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import socket
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from coach.config import CONFIG_PATH, load_config
from coach.engines import Engines
from coach.server import create_app
from coach.storage import Storage

DEFAULT_PORT = 8765


def free_port(preferred: int) -> int:
    """The preferred port if it's free, otherwise any free one (so a second
    copy, or another program on 8765, doesn't stop the app from starting)."""
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free port")


def start_server(port: int, cfg: dict) -> tuple[uvicorn.Server, threading.Thread]:
    app = create_app(cfg, Engines(cfg), Storage())
    # 127.0.0.1 only: the app is private to this computer.
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:        # wait until it accepts connections
        time.sleep(0.05)
    return server, thread


def open_window(url: str) -> bool:
    """Show the app in a native window. Returns False if pywebview can't run."""
    try:
        import webview
    except ImportError:
        return False
    # The Qt backend (pip-installable) can grant microphone access to the page;
    # see the README for why GTK isn't used on this setup.
    os.environ.setdefault("PYWEBVIEW_GUI", "qt")
    try:
        webview.create_window("Greenroom", url, width=1100, height=800, min_size=(380, 560))
        webview.start(gui=os.environ["PYWEBVIEW_GUI"], private_mode=False)
        return True
    except Exception as exc:  # no GUI toolkit, no display, ...
        print(f"Desktop window unavailable ({exc}); opening the browser instead.")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Greenroom: offline English speaking coach")
    ap.add_argument("--browser", action="store_true", help="open in the default browser, not a window")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--config", default=str(CONFIG_PATH), help="path to config.json")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    port = free_port(args.port)
    server, thread = start_server(port, cfg)
    url = f"http://127.0.0.1:{port}/"
    print(f"Greenroom is running at {url}", flush=True)

    if args.browser or not open_window(url):
        webbrowser.open(url)
        print("Press Ctrl+C to quit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    # Stop the server and give it time to save any open conversation
    # (the server's shutdown step does that), instead of killing it mid-write.
    server.should_exit = True
    thread.join(timeout=15)


if __name__ == "__main__":
    main()
