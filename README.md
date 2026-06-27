# Crystal

A local web app for cleaning up recorded dialogue. Import one mic or many, clean
each track, or combine multiple mics into one clear voice. FastAPI + WaveSurfer.js,
fully local, CPU-only, all open-source backends (noisereduce, DeepFilterNet3,
RNNoise).

## Run
```bash
./run.sh            # or: .venv/bin/python -m uvicorn server:app --port 7860
# open http://127.0.0.1:7860
```
Drag-drop a folder or WAV files onto the page, or use **Choose files / Choose
folder**. Works with a single file or a multi-mic recording.

## Flows
- **Clean tracks** — each track gets its own cleanup chain. `Render` the focused
  track, `⇧R` render all (per-track failures isolated).
- **Merge mics** — combine multiple mics of the same take into one mono voice.
  Pick a combine method, manage which mics are used, optionally clean each mic
  first.

## Cleanup chain
An ordered rack of stages — add / remove / reorder (↑↓):
- **noisereduce** — spectral gating. `stationary` toggle, `strength`.
- **DeepFilterNet** — DFN3 AI speech enhancement. `max atten (dB)` (0 = unlimited;
  raise ~12–18 to keep it natural / protect transients), `dry/wet mix`
  (lower re-adds original to mask artifacts on distant/quiet voice).
- **RNNoise** — `arnndn` + trained model. `model` (5 in `models/`), `wet/dry mix`.

Stages run top→bottom. Each track keeps its own chain; Merge has its own.

## Combining mics (Merge)
- **Blend** — weighted sum of all selected mics. Per time-frequency cell each mic
  is weighted by local SNR, biased toward speech-like content. Attenuates noise
  but always includes some of every mic.
- **Auto-pick best mic** — selects the single cleanest mic per moment (~46 ms
  windows, hysteresis-smoothed, crossfaded, level-matched). A mic with handling
  noise / wind / thumps isn't selected while it's bad, so that noise is excluded
  rather than blended in. An **avoid-noisy-mic sensitivity** dial controls how
  strongly noisy mics are skipped.
- **Mic panel** — include / exclude / **Solo** any mic; optional **auto-skip
  duplicate tracks**; **clean each mic before fusing** runs the chain per-source.

## Noise highlighting
**⚠ highlight noise** marks detected handling / low-frequency-noise regions on the
focused track's waveform (energy below the voice band + unvoiced speech band). A
heuristic — useful to see which mic is noisy when, then exclude/Solo accordingly.

## Shortcuts
space play · `[ ]` zoom · `← →` seek · `i o` trim in/out · `r` render ·
`⇧R` render all · `n p` track · `m` merge · `a` A/B · `e` export · `?` help.

## Output
`output/<normal|merge>/<name>.wav` (24-bit / 48 kHz). Export copies to
`output/export/`. Input is resampled to 48 kHz internally (required by DFN/RNNoise).
