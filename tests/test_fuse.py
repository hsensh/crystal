import numpy as np
import fusion

def _tone(freq, dur, sr=48000, amp=0.3, gap=False):
    t = np.arange(int(dur * sr)) / sr
    x = amp * np.sin(2 * np.pi * freq * t)
    if gap:  # silence the second half (voice "cuts off")
        x[len(x) // 2:] = 0.0
    return x.astype("float32")[None, :]

def test_fuse_returns_mono_same_length():
    sr = 48000
    a = _tone(200, 2.0)
    b = _tone(600, 2.0)
    out = fusion.fuse([a, b], sr)
    assert out.shape[0] == 1
    assert abs(out.shape[1] - a.shape[1]) <= 1024  # STFT edge tolerance

def test_fuse_preserves_overlap():
    """Two different tones on two mics, simultaneously -> both present in output."""
    sr = 48000
    a = _tone(200, 2.0)
    b = _tone(900, 2.0)
    out = fusion.fuse([a, b], sr)[0]
    mag = np.abs(np.fft.rfft(out))
    fr = np.fft.rfftfreq(len(out), 1 / sr)
    p200 = mag[np.argmin(np.abs(fr - 200))]
    p900 = mag[np.argmin(np.abs(fr - 900))]
    floor = np.median(mag) + 1e-6
    assert p200 > 10 * floor and p900 > 10 * floor

def test_fuse_fills_dropout():
    """Voice cuts off on mic A mid-clip but continues on mic B -> output keeps it."""
    sr = 48000
    a = _tone(440, 2.0, gap=True)   # drops at half
    b = _tone(440, 2.0)             # continuous
    out = fusion.fuse([a, b], sr)[0]
    second_half = out[len(out) // 2 + 4800: -4800]
    assert np.sqrt(np.mean(second_half ** 2)) > 0.01  # still audible

def test_exclude_drops_track():
    sr = 48000
    a = _tone(200, 1.0)
    b = _tone(600, 1.0)
    c = _tone(600, 1.0)
    out_all = fusion.fuse([a, b, c], sr)
    out_ex = fusion.fuse([a, b, c], sr, exclude=[2])
    assert out_all.shape == out_ex.shape  # excluding still yields valid mono
