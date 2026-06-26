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
    """Quality-aware gain-share auto-mixer. Sample-locked inputs (no realignment).

    Per STFT cell, weight each mic by local SNR, then modulate by how
    speech-like the frame is so loud non-speech (mic rubbing, handling, wind)
    does NOT win just for being loud:
      * spectral-flatness penalty — speech is harmonic/peaky (low flatness);
        rubbing/wind is broadband (high flatness) and gets down-weighted.
      * transient/handling guard — frames whose energy is an extreme outlier
        above the mic's own loud baseline (sudden bursts) are scaled down.
      * digital-clip guard — frames railed at full scale are down-weighted.
    Normalize across mics, sum complex spectra. Still preserves simultaneous
    speakers (each on its own mic) and fills dropouts.
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
        mag = np.abs(Z)
        mag2 = (mag ** 2) + 1e-12                      # (F, T)
        # per-bin noise floor = 10th percentile magnitude² over time
        floor = np.percentile(mag2, 10, axis=1, keepdims=True) + 1e-12
        w = mag2 / floor                              # local SNR-ish weight

        # per-frame quality factor (T,) modulates the whole frame
        q = _frame_quality(mag, mag2)
        # digital-clip guard: frames railed at full scale
        rail = _railed_frame_mask(m, Z.shape[1])      # (T,)
        q = q * (1.0 - 0.9 * rail)

        w = w * q[None, :]
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


def _frame_quality(mag, mag2, flat_floor=0.25, transient_k=4.0):
    """Per-frame (T,) 0..1 weight: high for speech-like frames, low for
    broadband or sudden-burst (handling/rubbing) frames.

    flatness = geo-mean / arith-mean of the frame magnitude spectrum
    (≈0 tonal/voiced, ≈1 broadband noise). speechiness = (1 - flatness),
    floored so broadband consonants (s/f) aren't gutted.
    transient: only frames far above the mic's own 90th-pct energy are scaled
    (so loud-but-steady speech is untouched, brief loud bursts are not).
    """
    eps = 1e-12
    log_gmean = np.exp(np.mean(np.log(mag + eps), axis=0))   # (T,)
    amean = np.mean(mag, axis=0) + eps                       # (T,)
    flatness = np.clip(log_gmean / amean, 0.0, 1.0)
    speechiness = np.maximum(1.0 - flatness, flat_floor)

    e = mag2.sum(axis=0)                                     # frame energy (T,)
    ref = np.percentile(e, 90) + eps
    ratio = e / ref
    transient = np.where(ratio > transient_k, transient_k / ratio, 1.0)

    return (speechiness * transient).astype("float32")

def autopick(audios, sr=48000, exclude=None, win_ms=46.0, smooth_ms=120.0):
    """Automatic best-mic mixer: pick the cleanest mic per moment, gate the rest,
    crossfade switches, level-match. Unlike fuse() this does NOT sum every mic —
    a mic with rubbing/handling simply isn't selected while it's bad, so the
    artifact is excluded (not just attenuated). Output stays leveled because mics
    are RMS-matched and the per-mic gain envelopes sum to 1.

    Per window: score each mic by SNR x speech-likeness x transient/clip guards;
    softly favor the winner; smooth the choice over time to avoid chattering;
    upsample to a sample-rate gain envelope and crossfade.
    """
    exclude = set(exclude or [])
    mics = [x for i, x in enumerate(audios) if i not in exclude]
    if not mics:
        raise ValueError("no mics left after exclude")
    monos = [x.mean(0).astype("float32") for x in mics]
    n = max(len(m) for m in monos)
    monos = [np.pad(m, (0, n - len(m))) for m in monos]

    # level-match: scale each mic to a common RMS so switching doesn't jump level
    rmss = [np.sqrt(np.mean(m ** 2)) + 1e-9 for m in monos]
    target = float(np.median(rmss))
    monos = [m * (target / r) for m, r in zip(monos, rmss)]

    hop = max(1, int(sr * win_ms / 1000))
    nfr = n // hop
    if nfr < 2:
        return (sum(monos) / len(monos))[None, :n].astype("float32")

    # per-mic per-frame quality score via STFT-derived metric.
    # score = speech-likeness x "has signal" — loudness is SATURATED so a LOUD
    # dirty mic (rubbing burst) can't win on energy; clean speech-like wins.
    scores = []
    for m in monos:
        _, _, Z = stft(m, fs=sr, nperseg=_NFFT, noverlap=_NFFT - _HOP,
                       boundary=None, padded=True)
        mag = np.abs(Z); mag2 = mag ** 2 + 1e-12
        floor = np.percentile(mag2, 10, axis=1, keepdims=True) + 1e-12
        snr = np.mean(mag2 / floor, axis=0)               # (T_stft,)
        ref = np.median(snr) + 1e-9
        has_signal = snr / (snr + ref)                    # 0..1, saturates loudness
        q = _frame_quality(mag, mag2)                     # speechiness x transient guard
        s = q * has_signal
        # resample stft-frame score to our window grid
        xp = np.linspace(0, 1, len(s)); xq = np.linspace(0, 1, nfr)
        scores.append(np.interp(xq, xp, s))
    S = np.stack(scores)                                   # (M, nfr)

    # HARD pick the best mic per frame, then apply hysteresis (median filter the
    # winner sequence) so isolated noisy frames don't flip the choice — only a
    # sustained better mic switches. Loser -> exactly 0 (no leakage).
    winner = np.argmax(S, axis=0)                          # (nfr,)
    k = max(1, int(round(smooth_ms / win_ms)))
    if k >= 3:
        winner = _median_filter_int(winner, k if k % 2 else k + 1)
    W = np.zeros_like(S)
    W[winner, np.arange(nfr)] = 1.0
    if k > 1:                                             # moving-average = crossfade
        ker = np.ones(k) / k
        W = np.stack([np.convolve(w, ker, mode="same") for w in W])
        W /= (W.sum(0, keepdims=True) + 1e-9)

    # upsample envelopes to sample rate (linear) and mix
    t_fr = (np.arange(nfr) + 0.5) * hop
    t_s = np.arange(n)
    out = np.zeros(n, dtype="float32")
    for m, w in zip(monos, W):
        env = np.interp(t_s, t_fr, w).astype("float32")
        out += m * env

    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0.99:
        out *= 0.99 / peak
    return out[None, :].astype("float32")


def _median_filter_int(seq, k):
    """Median filter an integer sequence (winner indices) with window k (odd).
    Smooths out isolated single-frame flips = hysteresis on mic choice."""
    half = k // 2
    pad = np.pad(seq, half, mode="edge")
    return np.array([int(np.median(pad[i:i + k])) for i in range(len(seq))])


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
