"""Desktop launcher for Dialogue Cleaner.

Runs the FastAPI server in a background thread and shows it in a native window
(pywebview) — no browser, no terminal. On first launch it makes sure the heavy
optional backend (torch / torchaudio / DeepFilterNet) is installed; the
DeepFilterNet model itself downloads to a user cache on first use.

Run in dev:  .venv/bin/python desktop.py
Packaged:    see BUILD.md (PyInstaller, per-OS).
"""
import os
import socket
import sys
import threading
import time
import urllib.request


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _serve(port):
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=port, log_level="warning")


def _wait_up(url, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    return False


def main():
    # run from the app's own directory so server:app + frontend/ resolve
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import resources
    resources.add_site_to_path()  # first-run-installed deps become importable
    port = _free_port()
    threading.Thread(target=_serve, args=(port,), daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    if not _wait_up(url):
        print("server failed to start", file=sys.stderr)
        sys.exit(1)
    import webview
    webview.create_window("Dialogue Cleaner", url, width=1280, height=820, min_size=(900, 600))
    webview.start()


if __name__ == "__main__":
    main()
