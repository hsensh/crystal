# Film dialogue denoise — comparison UI

Local Gradio app to bake off open-source denoise methods on the shoot tracks,
tune params live, A/B compare, and save outputs. All free / open-source, CPU-only.

## Run
```bash
.venv/bin/python app.py
# opens http://127.0.0.1:7860
```

## Tracks (`files/`)
4× 48 kHz / 24-bit, 121 s. `Tr1`, `Tr3`, `TrA` = mono mics; `TrLR` = stereo.

## Methods (your tool list, Demucs excluded)
| Method | Pipeline | Notes |
|---|---|---|
| `noisereduce` | spectral gating | surgical; auto noise-estimate (non-stationary if box off) |
| `deepfilternet` | DFN3 AI speech enhance | best on wind/traffic/crowd; cap attenuation to spare teacup |
| `rnnoise` | ffmpeg `arnndn` + trained model | light RNN; wet/dry mix keeps transients |
| `noisereduce__deepfilternet` | gate → AI | the recommended stack |
| `rnnoise__deepfilternet` | RNN → AI | |
| `noisereduce__rnnoise` | gate → RNN | |

## Params
- **noisereduce**: `stationary` toggle, `strength` (prop_decrease).
- **deepfilternet**: `max attenuation dB` — `0` = unlimited (most aggressive);
  raise (e.g. 12–24) to protect the teacup clack / transients.
- **rnnoise**: model choice (5 community models in `models/`), `wet/dry mix`
  (lower = more original preserved).

## Output
Each Render saves `output/<method>/<track>.wav` (24-bit / 48 kHz, streaming-ready).
Stats shown per render: peak / RMS / noise-floor dBFS (lower noise floor = more
suppression; watch RMS doesn't drop = dialogue intact).

## Tuning tips for your case (keep teacup, kill wind/cars)
- Start `deepfilternet` with attenuation 12–18 dB → natural, transient-safe.
- For max clean dialogue: `noisereduce__deepfilternet`.
- If a method eats the teacup, lower `rnnoise` mix or raise DFN attenuation cap.
