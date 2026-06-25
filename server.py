"""FastAPI backend for the dialogue cleaner. Serves the static frontend and
JSON endpoints that drive processing.py + fusion.py."""
import os
import tempfile

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import fusion

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
