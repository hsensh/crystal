"""FastAPI backend for the dialogue cleaner. Serves the static frontend and
JSON endpoints that drive processing.py + fusion.py."""
import os
import shutil
import tempfile

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import fusion
import processing as P

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(ROOT, "frontend")
SAFE_ROOTS = (ROOT, tempfile.gettempdir(), os.path.realpath(tempfile.gettempdir()))

app = FastAPI(title="Dialogue Cleaner")


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


@app.post("/api/session")
def session(req: SessionReq):
    return fusion.scan_session(req.paths)


@app.get("/api/audio")
def audio(path: str = Query(...)):
    return FileResponse(_safe(path), media_type="audio/wav")


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
    method: str
    params: dict = {}
    trim: list[float] = [0.0, 0.0]
    mode: str = "normal"


def _process_one(req: RenderReq):
    audio, sr = P.load(_safe(req.path))
    audio, sr = P.resample(audio, sr)
    audio = P.trim(audio, sr, req.trim[0], req.trim[1])
    before = P.stats(audio)
    out = P.run(req.method, audio, sr, **req.params)
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
    method: str = "deepfilternet"
    params: dict = {}
    trim: list[float] = [0.0, 0.0]


@app.post("/api/merge")
def merge(req: MergeReq):
    audios, sr = [], None
    for p in req.paths:
        a, s = P.load(_safe(p))
        a, s = P.resample(a, s)
        audios.append(a)
        sr = s
    if req.exclude:
        exclude = req.exclude
    else:
        auto = fusion.detect_derived(audios)
        # guard: never exclude all tracks
        exclude = auto if len(auto) < len(audios) else []
    fused = fusion.fuse(audios, sr, exclude=exclude)
    fused = P.trim(fused, sr, req.trim[0], req.trim[1])
    before = P.stats(fused)
    out = P.run(req.method, fused, sr, **req.params)
    after = P.stats(out)
    name = "merge.wav"
    out_path = os.path.join(OUT_DIR, "merge", name)
    P.save(out, sr, out_path)
    return {"name": name, "out_path": out_path, "before": before, "after": after,
            "excluded": exclude}


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
