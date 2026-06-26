import numpy as np
import processing as P


def _rms_db(a):
    return 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9)


def test_leveler_evens_loud_and_quiet_passages():
    """Leveler should raise a quiet passage toward a loud one (shrink the gap),
    so a near vs muffled speaker on one mic end up closer in loudness."""
    sr = 48000
    t = np.arange(int(4.0 * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 300 * t)
    x[len(x) // 2:] *= 0.1                      # second half much quieter
    x = x.astype("float32")[None, :]

    out = P.m_leveler(x, sr, lvl_max_gain_db=18, lvl_smooth_ms=400)
    h = out.shape[1] // 2
    gap_in = _rms_db(x[:, :h]) - _rms_db(x[:, h:])
    gap_out = _rms_db(out[:, :h]) - _rms_db(out[:, h:])
    assert gap_out < gap_in - 4.0              # gap meaningfully reduced


def test_leveler_in_chain_before_denoise():
    sr = 48000
    x = (0.2 * np.random.default_rng(0).standard_normal((1, sr))).astype("float32")
    out = P.run_chain(
        [{"type": "leveler", "params": {"lvl_max_gain_db": 12}},
         {"type": "noisereduce", "params": {"prop_decrease": 0.5}}],
        x, sr)
    assert out.shape[0] == 1
