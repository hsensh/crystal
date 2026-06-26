"""FastAPI backend for the dialogue cleaner. Serves the static frontend and
JSON endpoints that drive processing.py + fusion.py."""
import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import fusion
import processing as P

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(ROOT, "frontend")
FILES_DIR = os.path.join(ROOT, "files")
SAFE_ROOTS = (ROOT, tempfile.gettempdir(), os.path.realpath(tempfile.gettempdir()))
UPLOAD_ROOT = os.path.join(tempfile.gettempdir(), "dialogue-cleaner-uploads")

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


@app.get("/api/default_session")
def default_session():
    """Auto-load the project's files/ dir on startup so the user sees tracks
    immediately without typing a path."""
    if not os.path.isdir(FILES_DIR):
        return {"tracks": [], "mergeable": False, "reason": "no files/ dir"}
    return fusion.scan_session([FILES_DIR])


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
    auto_exclude: bool = True          # auto-drop detected mixdowns when no manual exclude
    preclean: bool = True              # clean each mic BEFORE fusing (per-source)
    fuse_mode: str = "blend"           # "blend" (sum all) | "autopick" (best mic per moment)
    chain: list[dict] = []             # ordered [{type, params}]; empty = passthrough
    trim: list[float] = [0.0, 0.0]


def _combine(audios, sr, mode):
    return fusion.autopick(audios, sr) if mode == "autopick" else fusion.fuse(audios, sr)


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
    raw = P.trim(_combine(active, sr, req.fuse_mode), sr, req.trim[0], req.trim[1])
    before = P.stats(raw)

    if req.preclean and req.chain:
        # clean each mic on its own (voice-specific) THEN combine the clean mics —
        # removes rubbing/noise per source before it gets summed/selected
        cleaned = [_denoise(a, sr, req.chain) for a in active]
        out = _combine(cleaned, sr, req.fuse_mode)
        out = P.trim(out, sr, req.trim[0], req.trim[1])
    else:
        # combine raw, then clean the result
        out = _denoise(raw, sr, req.chain) if req.chain else raw

    after = P.stats(out)
    name = "merge.wav"
    out_path = os.path.join(OUT_DIR, "merge", name)
    P.save(out, sr, out_path)
    return {"name": name, "out_path": out_path, "before": before, "after": after,
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
