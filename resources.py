"""First-run resource management for the packaged app.

The installer ships small (no torch). On first run the app downloads/installs the
heavy backend into a per-user app-support dir and shows a download screen:
  - torch / torchaudio / deepfilternet  (pip, into <appsupport>/site)
  - the DeepFilterNet3 model checkpoint (downloads to its own cache on first use)
ffmpeg is bundled with the app at build time (CI); in dev it comes from PATH.

Everything here is import-safe with no heavy deps, so it can run before torch
exists. Heavy installs go to a writable site dir added to sys.path.
"""
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import threading

APP = "DialogueCleaner"
HEAVY = ["torch", "torchaudio", "deepfilternet"]


def app_support():
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    elif os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    d = os.path.join(base, APP)
    os.makedirs(d, exist_ok=True)
    return d


def site_dir():
    d = os.path.join(app_support(), "site")
    os.makedirs(d, exist_ok=True)
    return d


def add_site_to_path():
    """Make any first-run-installed packages importable. Call at startup."""
    d = site_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    importlib.invalidate_caches()


def _frozen_base():
    return getattr(sys, "_MEIPASS", None)


def ffmpeg_path():
    """Bundled ffmpeg (frozen) or PATH (dev)."""
    base = _frozen_base()
    if base:
        exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        cand = os.path.join(base, exe)
        if os.path.isfile(cand):
            return cand
    return shutil.which("ffmpeg") or "ffmpeg"


def _have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:  # noqa: BLE001
        return False


def have_ffmpeg():
    p = ffmpeg_path()
    return os.path.isfile(p) or shutil.which(p) is not None


def status():
    add_site_to_path()
    torch = _have("torch") and _have("torchaudio")
    dfn = _have("df")  # deepfilternet imports as `df`
    return {
        "ffmpeg": have_ffmpeg(),
        "torch": torch,
        "deepfilternet": dfn,
        "ready": torch and dfn and have_ffmpeg(),
    }


# ---- install (threaded, with progress state the UI polls) ----

_state = {"running": False, "done": False, "ok": False, "log": [], "error": None}


def install_state():
    return dict(_state, log=_state["log"][-12:])


def _log(msg):
    _state["log"].append(msg)


def _uv():
    base = _frozen_base()
    if base:
        exe = "uv.exe" if os.name == "nt" else "uv"
        cand = os.path.join(base, exe)
        if os.path.isfile(cand):
            return cand
    return shutil.which("uv")


PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}"


def _pip_install(pkgs, target):
    """Install pkgs into target dir. Prefer bundled uv (works in frozen apps that
    have no pip — uv fetches a matching managed Python to resolve against); fall
    back to the current interpreter's pip (dev)."""
    uv = _uv()
    frozen = _frozen_base() is not None
    if uv and frozen:
        # frozen: sys.executable is the app, not python — let uv use a managed
        # Python matching the app's version so the wheels are ABI-compatible
        cmd = [uv, "pip", "install", "--python", PY_VER, "--target", target, *pkgs]
    elif uv:
        cmd = [uv, "pip", "install", "--python", sys.executable, "--target", target, *pkgs]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--target", target, *pkgs]
    _log("running " + os.path.basename(cmd[0]) + " pip install …")
    subprocess.run(cmd, check=True)


def _do_install():
    try:
        target = site_dir()
        missing = [m for m in HEAVY if not _have(m if m != "deepfilternet" else "df")]
        if missing:
            _log(f"Installing {', '.join(missing)} (large, one-time)…")
            _pip_install(missing, target)
            add_site_to_path()
        _log("Fetching DeepFilterNet model…")
        try:
            _warm_dfn_model()
            _log("Model ready.")
        except Exception as e:  # noqa: BLE001
            _log(f"Model will download on first use ({e}).")
        _state["ok"] = status()["ready"]
        _log("Done." if _state["ok"] else "Some resources still missing.")
    except Exception as e:  # noqa: BLE001
        _state["error"] = str(e)
        _log("Error: " + str(e))
    finally:
        _state["done"] = True
        _state["running"] = False


def _warm_dfn_model():
    """Trigger the DeepFilterNet model download by initialising it once."""
    import processing as P
    P.m_deepfilternet  # noqa: B018 — ensure module import path works
    import torch  # noqa: F401
    P._shim_torchaudio_backend()
    from df.enhance import init_df
    init_df()  # downloads + caches the checkpoint


def start_install():
    if _state["running"]:
        return
    _state.update(running=True, done=False, ok=False, log=[], error=None)
    threading.Thread(target=_do_install, daemon=True).start()
