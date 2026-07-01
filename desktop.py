"""Desktop launcher for Crystal.

Runs the FastAPI server in a background thread and shows it in a native window
(pywebview). Heavy backend (torch / DeepFilterNet) installs on first run via the
in-app download screen; the DeepFilterNet model downloads to a user cache on
first use.

Run in dev:  .venv/bin/python desktop.py
Packaged:    see BUILD.md (PyInstaller, per-OS).
"""
import os
import socket
import sys
import threading
import time
import traceback
import urllib.request


def _log_path():
    try:
        import resources
        return os.path.join(resources.app_support(), "crash.log")
    except Exception:  # noqa: BLE001
        return os.path.join(os.path.expanduser("~"), "crystal-crash.log")


def _log(msg):
    try:
        with open(_log_path(), "a") as f:
            f.write(msg + "\n")
    except Exception:  # noqa: BLE001
        pass


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _serve(port):
    try:
        import uvicorn
        import server  # import the app object directly (frozen-safe)
        # Running in a background thread: uvicorn's default signal handlers only
        # work on the main thread and raise on Windows. Disable them explicitly.
        config = uvicorn.Config(server.app, host="127.0.0.1", port=port,
                                log_level="warning")
        srv = uvicorn.Server(config)
        srv.install_signal_handlers = lambda: None
        srv.run()
    except Exception:  # noqa: BLE001
        _log("server thread crashed:\n" + traceback.format_exc())


def _wait_up(url, timeout=40.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    return False


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        import resources
        resources.add_site_to_path()
    except Exception:  # noqa: BLE001
        _log("resources init failed:\n" + traceback.format_exc())

    port = _free_port()
    threading.Thread(target=_serve, args=(port,), daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    if not _wait_up(url):
        _log("server did not come up within timeout")
        sys.exit(1)

    # Prefer a native window; if the bundled webview backend is unavailable,
    # fall back to the default browser so the app still works (and log why).
    try:
        import webview
        webview.create_window("Crystal", url, width=1280, height=820, min_size=(900, 600))
        webview.start()
    except Exception:  # noqa: BLE001
        _log("pywebview unavailable, falling back to browser:\n" + traceback.format_exc())
        import webbrowser
        webbrowser.open(url)
        while True:                       # keep the server alive for the browser tab
            time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        _log("fatal:\n" + traceback.format_exc())
        raise
