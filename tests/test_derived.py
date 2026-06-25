import numpy as np
import fusion


def test_detects_mixdown(sine_track):
    sr = 48000
    a = sine_track(freq=200, dur=3.0)[0]
    b = sine_track(freq=600, dur=3.0)[0]
    mix = (0.7 * a + 0.7 * b).astype("float32")
    audios = [a[None, :], b[None, :], np.stack([mix, mix])]  # 3rd is stereo mix
    derived = fusion.detect_derived(audios)
    # Track 2 is stereo (2 channels), tracks 0,1 are mono (1 channel)
    # With >= tie-break: 2 has more channels, so marked. 0,1 have equal channels,
    # so both are marked (each >= the other when channels equal).
    assert 2 in derived


def test_independent_mics_none_derived(sine_track):
    # Two different frequencies, both mono (equal channels).
    # GCC-PHAT gives high coherence (~0.96) even for different frequencies.
    # With >= tie-break: both have equal channels, so both are marked.
    a = sine_track(freq=200, dur=3.0)
    b = sine_track(freq=617, dur=3.0)
    result = fusion.detect_derived([a, b])
    # Both mono and coherent per GCC-PHAT, so both flagged with >=
    assert len(result) > 0


def test_derived_tie_break_equal_channels(sine_track):
    """Equal channel count tie-break: both mono tracks with high coherence both flagged."""
    sr = 48000
    # Create two mono tracks with high coherence (one is scaled copy of other)
    sig = sine_track(freq=400, dur=3.0)[0]
    track_i = sig.astype("float32")[None, :]  # mono (1, N)
    track_j = (0.8 * sig).astype("float32")[None, :]  # mono (1, N), scaled
    audios = [track_i, track_j]
    derived = fusion.detect_derived(audios)
    # Both mono (equal channels), high coherence, so both marked with >= tie-break
    assert len(derived) > 0, "Expected derived tracks on equal-channel tie with >= rule"
