# Dialogue Cleaner

Local web app to import a recording's mic tracks and clean them — either
per-track or by fusing mics into one voice. FastAPI + WaveSurfer.js, fully
local, CPU-only, all open-source backends (noisereduce, DeepFilterNet3, RNNoise).

## Run
```bash
./run.sh            # or: .venv/bin/python -m uvicorn server:app --port 7860
# open http://127.0.0.1:7860
```
Your project's `files/` folder auto-loads on open. You can also drag-drop files
or pick a folder.

## Flows
- **Clean tracks (Normal)** — each track gets its own cleanup chain. `Render` the
  focused track, `⇧R` render all (per-track failures isolated).
- **Merge mics** — fuse selected mics into one mono master. A mic panel lets you
  include/exclude or **Solo** each mic; "clean each mic before fusing" runs the
  chain per-source first. Trim to a region, Solo the clean mic → that section
  comes only from that mic.

## Cleanup chain
Build an ordered rack of stages — add / remove / reorder (↑↓) any of:
- **noisereduce** — spectral gating. `stationary` toggle, `strength`.
- **DeepFilterNet** — DFN3 AI speech enhance. `max atten (dB)` (0 = unlimited;
  raise ~12–18 to protect transients), `dry/wet mix` (lower re-adds original to
  mask artifacts on distant/quiet voice).
- **RNNoise** — `arnndn` + trained model. `model` (5 in `models/`), `wet/dry mix`.

Stages run top→bottom. Each track keeps its own chain; Merge has its own.

## Fusion (how Merge combines mics)
Sample-locked mics, no realignment. Per STFT cell, each mic is weighted by local
SNR, modulated by a spectral-flatness penalty (favor harmonic speech over
broadband noise) and a transient guard (down-weight sudden bursts), then summed.
Note: fusion **attenuates** noise like rubbing/handling but cannot remove it —
blending always includes some of every mic. To eliminate rubbing, exclude or
Solo the offending mics in the mic panel.

## Shortcuts
space play · `[ ]` zoom · `← →` seek · `i o` trim in/out · `r` render ·
`⇧R` render all · `n p` track · `m` merge · `a` A/B · `e` export · `?` help.

## Tracks (`files/`)
4× 48 kHz / 24-bit, ~121 s. `Tr1`, `Tr3`, `TrA` = mono lavs; `TrLR` = stereo.

## Output
`output/<normal|merge>/<name>.wav` (24-bit / 48 kHz). Export copies to
`output/export/`.
