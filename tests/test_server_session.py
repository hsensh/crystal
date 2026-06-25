from fastapi.testclient import TestClient
import server


def test_session_endpoint(write_wav, sine_track):
    a = write_wav("a.wav", sine_track(dur=1.0))
    b = write_wav("b.wav", sine_track(dur=1.0, freq=330))
    c = TestClient(server.app)
    r = c.post("/api/session", json={"paths": [a, b]})
    assert r.status_code == 200
    body = r.json()
    assert body["mergeable"] is True
    assert {t["name"] for t in body["tracks"]} == {"a.wav", "b.wav"}


def test_audio_endpoint_streams(write_wav, sine_track):
    a = write_wav("a.wav", sine_track(dur=1.0))
    c = TestClient(server.app)
    r = c.get("/api/audio", params={"path": a})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")
    assert len(r.content) > 1000


def test_audio_endpoint_rejects_path_traversal():
    c = TestClient(server.app)
    r = c.get("/api/audio", params={"path": "/etc/passwd"})
    assert r.status_code == 403
    assert "path outside allowed roots" in r.json()["detail"]
