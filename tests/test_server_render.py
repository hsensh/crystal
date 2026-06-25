from fastapi.testclient import TestClient
import server


def test_render_noisereduce(write_wav, sine_track):
    a = write_wav("a.wav", sine_track(dur=1.0))
    c = TestClient(server.app)
    r = c.post("/api/render", json={
        "path": a, "method": "noisereduce",
        "params": {"stationary": True, "prop_decrease": 0.8},
        "trim": [0, 0], "mode": "normal",
    })
    assert r.status_code == 200
    b = r.json()
    assert b["out_path"].endswith(".wav")
    assert "rms_dbfs" in b["after"]


def test_render_all_isolates_failure(write_wav, sine_track):
    a = write_wav("a.wav", sine_track(dur=1.0))
    c = TestClient(server.app)
    r = c.post("/api/render_all", json={"tracks": [
        {"path": a, "method": "noisereduce", "params": {"prop_decrease": 0.5},
         "trim": [0, 0], "mode": "normal"},
        {"path": "/nope/missing.wav", "method": "noisereduce", "params": {},
         "trim": [0, 0], "mode": "normal"},
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
        "paths": [a, b], "exclude": [], "method": "noisereduce",
        "params": {"prop_decrease": 0.5}, "trim": [0, 0],
    })
    assert r.status_code == 200
    assert r.json()["out_path"].endswith(".wav")
