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


def _safe(path):
    rp = os.path.realpath(path)
    if not any(rp.startswith(os.path.realpath(r)) for r in SAFE_ROOTS):
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
    return FileResponse(os.path.join(FRONTEND, "index.html"))


if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
