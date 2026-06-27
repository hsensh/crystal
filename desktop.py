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
import subprocess
import sys
import threading
import time
import urllib.request

HEAVY = ["torch", "torchaudio", "deepfilternet"]


def _missing_heavy():
    import importlib.util
    return [m for m in HEAVY if importlib.util.find_spec(m.replace("-", "_")) is None
            and importlib.util.find_spec("df" if m == "deepfilternet" else m) is None]


def ensure_heavy(status=print):
    """Install torch/torchaudio/DeepFilterNet on first run if absent.

    Dev / venv builds: pip into the current interpreter. Frozen builds bundle a
    Python + pip and install into a writable app-support dir (see BUILD.md).
    """
    missing = _missing_heavy()
    if not missing:
        return True
    status(f"First run: installing {', '.join(missing)} (one-time, large download)…")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
        status("Install complete.")
        return True
    except Exception as e:  # noqa: BLE001 — surface, don't crash the app
        status(f"Could not auto-install ({e}). Leveler + noisereduce still work; "
               "DeepFilterNet needs torch.")
        return False


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
    ensure_heavy()
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
