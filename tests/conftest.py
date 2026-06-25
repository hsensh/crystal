import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sine_track():
    """Return (C,T) float32 sine. Used as a stand-in 'voice' on a mic."""
    def _make(freq=220.0, dur=1.0, sr=48000, amp=0.3, channels=1):
        t = np.arange(int(dur * sr)) / sr
        x = (amp * np.sin(2 * np.pi * freq * t)).astype("float32")
        return np.tile(x, (channels, 1))
    return _make


@pytest.fixture
def write_wav(tmp_path):
    """Write (C,T) audio to a wav under tmp_path, return its path string."""
    def _write(name, audio, sr=48000):
        p = tmp_path / name
        sf.write(p, audio.T, sr, subtype="PCM_24")
        return str(p)
    return _write
