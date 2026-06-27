"""FastAPI backend for the dialogue cleaner. Serves the static frontend and
JSON endpoints that drive processing.py + fusion.py."""
import logging
import os
import shutil
import tempfile
import traceback

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import fusion
import processing as P
import resources

resources.add_site_to_path()  # make any first-run-installed packages importable

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(ROOT, "frontend")
SAFE_ROOTS = (ROOT, tempfile.gettempdir(), os.path.realpath(tempfile.gettempdir()))
UPLOAD_ROOT = os.path.join(tempfile.gettempdir(), "dialogue-cleaner-uploads")
LOG_FILE = os.path.join(resources.app_support(), "server.log")

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("crystal")

app = FastAPI(title="Crystal")


def _friendly(exc: Exception) -> str:
    """Plain-language message for end users (raw detail still goes to the logs)."""
    t = f"{type(exc).__name__}: {exc}".lower()
    if "torch" in t or "no module named 'df'" in t or "deepfilternet" in t:
        return ("The speech-AI engine isn't ready yet. If this is a fresh install, "
                "let the first-run download finish, then try again.")
    if "ffmpeg" in t or ("returned non-zero" in t and "dynaudnorm" in t) or "arnndn" in t:
        return ("The audio engine (ffmpeg) couldn't run on this machine. "
                "Reinstalling the app usually fixes this.")
    if "init_df" in t or "checkpoint" in t or "download" in t or "urlopen" in t:
        return ("Couldn't download the DeepFilterNet model. Check your internet "
                "connection and try again.")
    if "sr()" in t or "expects" in t and "got" in t:
        return "There was a sample-rate problem with this audio. Try re-importing it."
    return "Something went wrong while processing. Open Logs for the technical details."


@app.exception_handler(Exception)
async def _on_error(request: Request, exc: Exception):
    """Log full traceback to file (the packaged app has no console) and return a
    friendly message for the UI plus the raw detail for the Logs viewer."""
    log.error("error on %s\n%s", request.url.path, traceback.format_exc())
    return JSONResponse(status_code=500, content={
        "message": _friendly(exc),
        "detail": f"{type(exc).__name__}: {exc}",
    })


@app.post("/api/logs/clear")
def logs_clear():
    """Wipe the log files so the next reproduction shows only fresh entries."""
    for f in (LOG_FILE, os.path.join(resources.app_support(), "crash.log")):
        try:
            open(f, "w").close()
        except Exception:  # noqa: BLE001
            pass
    log.info("logs cleared")
    return {"cleared": True}


@app.get("/api/logs")
def logs():
    """Recent server + startup log lines, for the in-app Logs viewer."""
    parts = []
    for f in (LOG_FILE, os.path.join(resources.app_support(), "crash.log")):
        if os.path.isfile(f):
            try:
                with open(f) as fh:
                    tail = "".join(fh.readlines()[-200:])
                parts.append(f"==== {os.path.basename(f)} ====\n{tail}")
            except Exception:  # noqa: BLE001
                pass
    return PlainTextResponse("\n\n".join(parts) or "(no logs yet)")


@app.middleware("http")
async def _no_cache(request, call_next):
    """Never cache the app shell/static — so code changes show on a normal
    refresh instead of needing a hard-reload."""
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


def _under(rp, root):
    """Check if rp is exactly root or strictly inside it (separator-aware)."""
    root = os.path.realpath(root)
    return rp == root or rp.startswith(root + os.sep)


def _safe(path):
    rp = os.path.realpath(path)
    if not any(_under(rp, r) for r in SAFE_ROOTS):
        raise HTTPException(403, "path outside allowed roots")
    if not os.path.isfile(rp):
        raise HTTPException(404, "file not found")
    return rp


class SessionReq(BaseModel):
    paths: list[str]


@app.get("/api/resources")
def resources_status():
    """Resource availability + any in-progress first-run install state."""
    return {**resources.status(), "install": resources.install_state()}


@app.post("/api/resources/install")
def resources_install():
    resources.start_install()
    return {"started": True}


@app.post("/api/session")
def session(req: SessionReq):
    return fusion.scan_session(req.paths)


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    """Accept dropped/picked files, store them in a fresh temp session dir,
    return the scanned session. Lets the UI work by drag-drop or folder pick."""
    import uuid
    sess_dir = os.path.join(UPLOAD_ROOT, uuid.uuid4().hex[:8])
    os.makedirs(sess_dir, exist_ok=True)
    paths = []
    for f in files:
        name = os.path.basename(f.filename or "track.wav")
        if not name.lower().endswith(".wav"):
            continue
        dest = os.path.join(sess_dir, name)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(dest)
    if not paths:
        raise HTTPException(400, "no .wav files in upload")
    return fusion.scan_session(paths)


@app.get("/api/audio")
def audio(path: str = Query(...)):
    return FileResponse(_safe(path), media_type="audio/wav")


@app.get("/api/noise")
def noise(path: str = Query(...), thresh: float = 0.45):
    """Detected handling / low-freq-noise regions for a track, for waveform overlay."""
    a, s = P.load(_safe(path))
    a, s = P.resample(a, s)
    segs = fusion.noise_segments(a.mean(0), s, thresh=thresh)
    return {"segments": segs}


@app.get("/")
def index():
    index_path = os.path.join(FRONTEND, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(404, "frontend not built")
    return FileResponse(index_path)


if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


OUT_DIR = os.path.join(ROOT, "output")


class RenderReq(BaseModel):
    path: str
    chain: list[dict] = []  # ordered [{type, params}]; empty = passthrough
    trim: list[float] = [0.0, 0.0]
    mode: str = "normal"


def _denoise(audio, sr, chain):
    return P.run_chain(chain, audio, sr)


def _process_one(req: RenderReq):
    audio, sr = P.load(_safe(req.path))
    audio, sr = P.resample(audio, sr)
    audio = P.trim(audio, sr, req.trim[0], req.trim[1])
    before = P.stats(audio)
    out = _denoise(audio, sr, req.chain)
    after = P.stats(out)
    name = os.path.basename(req.path)
    out_path = os.path.join(OUT_DIR, req.mode, name)
    P.save(out, sr, out_path)
    return {"name": name, "out_path": out_path, "before": before, "after": after}


@app.post("/api/render")
def render(req: RenderReq):
    return _process_one(req)


class RenderAllReq(BaseModel):
    tracks: list[RenderReq]


@app.post("/api/render_all")
def render_all(req: RenderAllReq):
    results = []
    for t in req.tracks:
        try:
            r = _process_one(t)
            results.append({"name": r["name"], "ok": True, "out_path": r["out_path"],
                            "before": r["before"], "after": r["after"]})
        except Exception as e:  # per-track isolation
            results.append({"name": os.path.basename(t.path), "ok": False,
                            "error": str(e)})
    return {"results": results}


class MergeReq(BaseModel):
    paths: list[str]
    exclude: list[int] = []
    auto_exclude: bool = False         # opt-in: auto-skip detected duplicate tracks
    preclean: bool = True              # clean each mic BEFORE fusing (per-source)
    fuse_mode: str = "blend"           # "blend" (sum all) | "autopick" (best mic per moment)
    noise_strength: float = 1.0        # autopick: how hard to avoid noisy mics (0..2)
    win_ms: float = 46.0               # autopick: decision window (how fast it can switch)
    smooth_ms: float = 120.0           # autopick: hysteresis + crossfade length
    chain: list[dict] = []             # ordered [{type, params}]; empty = passthrough
    trim: list[float] = [0.0, 0.0]


def _combine(audios, sr, req):
    if req.fuse_mode == "autopick":
        return fusion.autopick(audios, sr, noise_strength=req.noise_strength,
                               win_ms=req.win_ms, smooth_ms=req.smooth_ms)
    return fusion.fuse(audios, sr)


@app.post("/api/merge")
def merge(req: MergeReq):
    audios, sr = [], None
    for p in req.paths:
        a, s = P.load(_safe(p))
        a, s = P.resample(a, s)
        audios.append(a)
        sr = s

    # which mics are excluded: explicit user list wins; else auto-detected mixdowns
    if req.exclude:
        exclude = list(req.exclude)
    elif req.auto_exclude:
        auto = fusion.detect_derived(audios)
        exclude = auto if len(auto) < len(audios) else []
    else:
        exclude = []
    active_idx = [i for i in range(len(audios)) if i not in exclude]
    if not active_idx:
        raise HTTPException(400, "all mics excluded")
    active = [audios[i] for i in active_idx]

    # reference "before" = raw combine (so the metric reflects no cleaning)
    raw = P.trim(_combine(active, sr, req), sr, req.trim[0], req.trim[1])
    before = P.stats(raw)

    if req.preclean and req.chain:
        # clean each mic on its own (voice-specific) THEN combine the clean mics —
        # removes noise per source before it gets summed/selected
        cleaned = [_denoise(a, sr, req.chain) for a in active]
        out = _combine(cleaned, sr, req)
        out = P.trim(out, sr, req.trim[0], req.trim[1])
    else:
        # combine raw, then clean the result
        out = _denoise(raw, sr, req.chain) if req.chain else raw

    after = P.stats(out)
    name = "merge.wav"
    out_path = os.path.join(OUT_DIR, "merge", name)
    P.save(out, sr, out_path)
    # also save the raw (uncleaned) combine so the UI can A/B it
    raw_path = os.path.join(OUT_DIR, "merge", "merge_raw.wav")
    P.save(raw, sr, raw_path)
    return {"name": name, "out_path": out_path, "raw_path": raw_path,
            "before": before, "after": after,
            "excluded": exclude, "active": active_idx, "preclean": req.preclean,
            "fuse_mode": req.fuse_mode}


class ExportReq(BaseModel):
    src: str
    dest_dir: str


@app.post("/api/export")
def export(req: ExportReq):
    src = _safe(req.src)
    os.makedirs(req.dest_dir, exist_ok=True)
    dest = os.path.join(req.dest_dir, os.path.basename(src))
    shutil.copy2(src, dest)
    return {"dest": dest}
