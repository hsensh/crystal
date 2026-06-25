import fusion


def test_mergeable_when_same_sr_and_length(write_wav, sine_track):
    a = write_wav("a.wav", sine_track(dur=2.0))
    b = write_wav("b.wav", sine_track(dur=2.0, freq=330))
    s = fusion.scan_session([a, b])
    assert s["mergeable"] is True
    assert len(s["tracks"]) == 2
    assert s["tracks"][0]["sr"] == 48000


def test_not_mergeable_on_length_mismatch(write_wav, sine_track):
    a = write_wav("a.wav", sine_track(dur=2.0))
    b = write_wav("b.wav", sine_track(dur=5.0))
    s = fusion.scan_session([a, b])
    assert s["mergeable"] is False
    assert "length" in s["reason"].lower() or "duration" in s["reason"].lower()


def test_directory_expands_to_wavs(tmp_path, write_wav, sine_track):
    write_wav("a.wav", sine_track(dur=1.0))
    write_wav("b.wav", sine_track(dur=1.0))
    s = fusion.scan_session([str(tmp_path)])
    assert len(s["tracks"]) == 2
