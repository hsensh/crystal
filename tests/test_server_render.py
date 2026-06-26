from fastapi.testclient import TestClient
import server


def test_render_noisereduce(write_wav, sine_track):
    a = write_wav("a.wav", sine_track(dur=1.0))
    c = TestClient(server.app)
    r = c.post("/api/render", json={
        "path": a,
        "chain": [{"type": "noisereduce", "params": {"stationary": True, "prop_decrease": 0.8}}],
        "trim": [0, 0], "mode": "normal",
    })
    assert r.status_code == 200
    b = r.json()
    assert b["out_path"].endswith(".wav")
    assert "rms_dbfs" in b["after"]


def test_render_all_isolates_failure(write_wav, sine_track):
    a = write_wav("a.wav", sine_track(dur=1.0))
    c = TestClient(server.app)
    nr = [{"type": "noisereduce", "params": {"prop_decrease": 0.5}}]
    r = c.post("/api/render_all", json={"tracks": [
        {"path": a, "chain": nr, "trim": [0, 0], "mode": "normal"},
        {"path": "/nope/missing.wav", "chain": nr, "trim": [0, 0], "mode": "normal"},
    ]})
    assert r.status_code == 200
    res = r.json()["results"]
    oks = {x["name"]: x["ok"] for x in res}
    assert oks["a.wav"] is True
    assert any(not x["ok"] for x in res)


def test_merge_endpoint(write_wav, sine_track):
    a = write_wav("a.wav", sine_track(dur=1.0, freq=200))
    b = write_wav("b.wav", sine_track(dur=1.0, freq=600))
    c = TestClient(server.app)
    r = c.post("/api/merge", json={
        "paths": [a, b], "exclude": [],
        "chain": [{"type": "noisereduce", "params": {"prop_decrease": 0.5}}], "trim": [0, 0],
    })
    assert r.status_code == 200
    assert r.json()["out_path"].endswith(".wav")
    assert "excluded" in r.json()


def test_merge_auto_excludes_derived(write_wav):
    """Stereo mixdown track (high coherence with mono mic) is auto-excluded.

    Uses broadband noise so GCC-PHAT coherence is meaningful:
    - mic0 and mic1 are independent noise sources → low coherence
    - trlr is a stereo duplicate of mic0 → coherence ~1 with mic0 → detected derived
    """
    import numpy as np
    rng = np.random.default_rng(42)
    n = int(3.0 * 48000)
    mic0 = (rng.standard_normal(n) * 0.3).astype("float32").reshape(1, -1)
    mic1 = (rng.standard_normal(n) * 0.3).astype("float32").reshape(1, -1)
    mix_stereo = np.vstack([mic0[0], mic0[0]])  # stereo copy of mic0 = derived mix
    a = write_wav("mic0.wav", mic0)
    b = write_wav("mic1.wav", mic1)
    c_mix = write_wav("trlr.wav", mix_stereo)
    client = TestClient(server.app)
    r = client.post("/api/merge", json={
        "paths": [a, b, c_mix], "exclude": [], "auto_exclude": True,
        "chain": [{"type": "noisereduce", "params": {"prop_decrease": 0.5}}], "trim": [0, 0],
    })
    assert r.status_code == 200
    body = r.json()
    assert "excluded" in body
    assert len(body["excluded"]) > 0, "derived stereo mix was not auto-excluded"
    assert 2 in body["excluded"], f"expected index 2 excluded, got {body['excluded']}"
