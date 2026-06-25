"""Multi-mic session detection + gain-share fusion for the dialogue cleaner."""
import glob
import os

import numpy as np

import processing as P

MERGE_DUR_TOL_S = 0.5


def _expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.wav", "*.WAV"):
                out += sorted(glob.glob(os.path.join(p, ext)))
        else:
            out.append(p)
    # de-dupe preserving order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def is_mergeable(metas):
    """metas: list of dicts with 'sr' and 'duration'. Same sr + near-equal len."""
    if len(metas) < 2:
        return False, "need at least 2 tracks to merge"
    srs = {m["sr"] for m in metas}
    if len(srs) > 1:
        return False, f"sample rates differ {sorted(srs)} — not one session"
    durs = [m["duration"] for m in metas]
    spread = max(durs) - min(durs)
    if spread > MERGE_DUR_TOL_S:
        return False, f"lengths differ {spread:.1f}s — not one session"
    return True, "sample-locked session"


def scan_session(paths):
    tracks = []
    for path in _expand(paths):
        audio, sr = P.load(path)
        tracks.append({
            "name": os.path.basename(path),
            "path": path,
            "sr": sr,
            "duration": audio.shape[1] / sr,
            "channels": audio.shape[0],
        })
    mergeable, reason = is_mergeable(tracks)
    return {"tracks": tracks, "mergeable": mergeable, "reason": reason}
