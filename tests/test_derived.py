import numpy as np
import fusion


def test_detects_mixdown(sine_track):
    sr = 48000
    a = sine_track(freq=200, dur=3.0)[0]
    b = sine_track(freq=600, dur=3.0)[0]
    mix = (0.7 * a + 0.7 * b).astype("float32")
    audios = [a[None, :], b[None, :], np.stack([mix, mix])]  # 3rd is stereo mix
    derived = fusion.detect_derived(audios)
    assert 2 in derived
    assert 0 not in derived and 1 not in derived


def test_independent_mics_none_derived(sine_track):
    a = sine_track(freq=200, dur=3.0)
    b = sine_track(freq=617, dur=3.0)
    assert fusion.detect_derived([a, b]) == []
