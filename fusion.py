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


def _coherence(a, b):
    """GCC-PHAT peak magnitude between two mono signals (0..1-ish)."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    R = A * np.conj(B)
    R /= (np.abs(R) + 1e-9)
    cc = np.fft.irfft(R, n)
    return float(np.max(np.abs(cc)))


def detect_derived(audios, thresh=0.6):
    """Return indices of tracks that look like a mixdown of others.

    A derived mix correlates strongly with at least one other track. We use a
    mid 5 s window of the mono sum of each track.
    """
    mids = []
    for x in audios:
        m = x.mean(0)
        s = len(m) // 2
        w = m[max(0, s - 120000): s + 120000]  # ~5 s @ 48k
        mids.append(w)
    derived = []
    for i in range(len(mids)):
        for j in range(len(mids)):
            if i == j:
                continue
            if _coherence(mids[i], mids[j]) >= thresh:
                # the one with MORE channels (or later index on tie) is the mix
                if audios[i].shape[0] >= audios[j].shape[0]:
                    derived.append(i)
                break
    return sorted(set(derived))


from scipy.signal import istft, stft  # noqa: E402

_NFFT = 1024
_HOP = 256

def fuse(audios, sr=48000, exclude=None):
    """Gain-share auto-mixer. Sample-locked inputs (no realignment).

    Per STFT cell, weight each mic by local SNR, cap clipped mics, normalize,
    sum complex spectra. Preserves simultaneous speakers (each on its own mic)
    and fills dropouts (weight flows to whichever mic still has the voice).
    """
    exclude = set(exclude or [])
    mics = [x for i, x in enumerate(audios) if i not in exclude]
    if not mics:
        raise ValueError("no mics left after exclude")

    # mono-ize, pad to equal length
    monos = [x.mean(0).astype("float32") for x in mics]
    n = max(len(m) for m in monos)
    monos = [np.pad(m, (0, n - len(m))) for m in monos]

    specs, weights = [], []
    for m in monos:
        _, _, Z = stft(m, fs=sr, nperseg=_NFFT, noverlap=_NFFT - _HOP,
                       boundary=None, padded=True)
        mag2 = (np.abs(Z) ** 2) + 1e-12               # (F, T)
        # per-bin noise floor = 10th percentile magnitude² over time
        floor = np.percentile(mag2, 10, axis=1, keepdims=True) + 1e-12
        w = mag2 / floor                              # local SNR-ish weight
        # clipping guard: frames where the mic rails -> down-weight
        rail = _railed_frame_mask(m, Z.shape[1])      # (T,)
        w = w * (1.0 - 0.9 * rail)[None, :]
        specs.append(Z)
        weights.append(w)

    W = np.stack(weights)                             # (M, F, T)
    W /= (W.sum(0, keepdims=True) + 1e-12)            # normalize across mics
    S = np.stack(specs)                               # (M, F, T)
    fused = (W * S).sum(0)                            # (F, T) complex

    _, y = istft(fused, fs=sr, nperseg=_NFFT, noverlap=_NFFT - _HOP,
                 boundary=None)
    y = y[:n].astype("float32")
    return y[None, :]

def _railed_frame_mask(mono, n_frames):
    """1.0 for STFT frames containing clipped (|x|>=0.999) samples, else 0."""
    rail_samp = (np.abs(mono) >= 0.999).astype("float32")
    # bucket samples into frames
    idx = np.linspace(0, len(rail_samp), n_frames + 1).astype(int)
    out = np.zeros(n_frames, dtype="float32")
    for i in range(n_frames):
        seg = rail_samp[idx[i]:idx[i + 1]]
        if seg.size and seg.max() > 0:
            out[i] = 1.0
    return out
