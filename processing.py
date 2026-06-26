"""Audio denoise backends + method dispatch for the comparison UI.

All methods take/return float32 audio as a 2D array shaped (channels, samples)
at a fixed sample rate (48 kHz for the film tracks). Keeps everything in-memory
so stacked pipelines just chain calls.
"""
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

SR = 48000
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RNN_MODELS = {
    "marathon-prescription (general, balanced)": "mp.rnnn",
    "beguiling-drafter (general)": "bd.rnnn",
    "somnolent-hogwash (general)": "sh.rnnn",
    "leavened-quisling (general)": "lq.rnnn",
    "conjoined-burgers (general)": "cb.rnnn",
}

# ---------- io helpers ----------

def load(path):
    """Read wav -> (audio (C,T) float32, sr)."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    return data.T.copy(), sr  # (C, T)


def resample(audio, sr, target=SR):
    """Resample (C,T) float32 to target sr. Gradio's editor re-encodes trimmed
    clips at 44.1k; DFN and the rnnoise models require 48k."""
    if sr == target:
        return audio, sr
    import torch
    import torchaudio.functional as AF
    t = torch.from_numpy(audio)
    out = AF.resample(t, sr, target).numpy().astype("float32")
    return out, target


def save(audio, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio.T, sr, subtype="PCM_24")  # write (T, C)


def trim(audio, sr, start_s=0.0, end_s=0.0):
    """Slice audio to [start_s, end_s) seconds. end_s<=0 or beyond = to the end."""
    n = audio.shape[1]
    a = max(0, int(float(start_s) * sr))
    b = n if float(end_s) <= 0 else min(n, int(float(end_s) * sr))
    if b <= a:
        b = n
    return audio[:, a:b]


def write_temp(audio, sr):
    """Write audio to a temp wav, return path (for UI players)."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio.T, sr, subtype="PCM_24")
    return path


def stats(audio):
    """Return dict of peak/RMS dBFS and an estimated noise floor (quiet 10%)."""
    x = audio.reshape(-1)
    peak = np.max(np.abs(x)) + 1e-12
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    # frame RMS, take 10th percentile as noise-floor proxy
    fr = 4800  # 100 ms
    n = (len(x) // fr) * fr
    frames = x[:n].reshape(-1, fr)
    frms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-12
    floor = np.percentile(frms, 10)
    db = lambda v: 20 * np.log10(v)
    return {"peak_dbfs": db(peak), "rms_dbfs": db(rms), "noise_floor_dbfs": db(floor)}


# ---------- single methods ----------

def m_noisereduce(audio, sr, stationary=True, prop_decrease=0.8, **_):
    import noisereduce as nr
    out = np.empty_like(audio)
    for c in range(audio.shape[0]):
        out[c] = nr.reduce_noise(
            y=audio[c], sr=sr, stationary=stationary,
            prop_decrease=float(prop_decrease),
        ).astype(np.float32)
    return out


_DF = None  # lazy-cached (model, state)

def _shim_torchaudio_backend():
    """Newer torchaudio dropped torchaudio.backend.common; DeepFilterNet's df.io
    still imports AudioMetaData from it. We never use df.io's file IO (we pass
    tensors straight to enhance), so a stub satisfies the import."""
    import sys
    import types
    if "torchaudio.backend.common" in sys.modules:
        return
    import torchaudio
    meta = getattr(torchaudio, "AudioMetaData", None)
    if meta is None:
        class meta:  # minimal stub
            pass
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = meta
    backend.common = common
    sys.modules.setdefault("torchaudio.backend", backend)
    sys.modules["torchaudio.backend.common"] = common


def m_deepfilternet(audio, sr, atten_lim_db=0.0, dfn_mix=1.0, **_):
    global _DF
    import torch
    _shim_torchaudio_backend()
    from df.enhance import init_df, enhance
    if _DF is None:
        model, state, _ = init_df()
        _DF = (model, state)
    model, state = _DF
    assert sr == state.sr(), f"DFN expects {state.sr()}, got {sr}"
    t = torch.from_numpy(audio)  # (C, T)
    lim = None if float(atten_lim_db) <= 0 else float(atten_lim_db)
    out = enhance(model, state, t, atten_lim_db=lim).cpu().numpy().astype(np.float32)
    # wet/dry blend: re-add original to mask musical-noise artifacts on
    # low-SNR (distant) speech. mix=1.0 -> full DFN, lower -> more original.
    mix = float(dfn_mix)
    if mix >= 1.0:
        return out
    n = min(out.shape[1], audio.shape[1])
    return (mix * out[:, :n] + (1 - mix) * audio[:, :n]).astype(np.float32)


def m_rnnoise(audio, sr, rnn_model="mp.rnnn", rnn_mix=1.0, **_):
    model_path = os.path.join(MODELS_DIR, rnn_model)
    with tempfile.TemporaryDirectory() as d:
        ip = os.path.join(d, "in.wav")
        op = os.path.join(d, "out.wav")
        sf.write(ip, audio.T, sr, subtype="FLOAT")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", ip,
             "-af", f"arnndn=m={model_path}", op],
            check=True,
        )
        wet, _ = sf.read(op, dtype="float32", always_2d=True)
    wet = wet.T  # (C, T)
    # length guard
    n = min(wet.shape[1], audio.shape[1])
    wet, dry = wet[:, :n], audio[:, :n]
    mix = float(rnn_mix)
    return (mix * wet + (1 - mix) * dry).astype(np.float32)


# ---------- user-ordered chain ----------

CHAIN_FNS = {
    "noisereduce": m_noisereduce,
    "deepfilternet": m_deepfilternet,
    "rnnoise": m_rnnoise,
}


def run_chain(stages, audio, sr):
    """Run an ordered list of stages on the audio.

    stages: list of {"type": <one of CHAIN_FNS>, "params": {...}}. Each stage's
    params are passed as kwargs to its backend (extras ignored via **_). Stages
    apply in list order, so the UI controls order + which methods are present.
    """
    for st in stages:
        fn = CHAIN_FNS[st["type"]]
        audio = fn(audio, sr, **st.get("params", {}))
    return audio
