# Dialogue Cleaner — production import-and-clean tool

**Date:** 2026-06-25
**Status:** Approved design

## Purpose

Primary production tool to import the audio tracks from one recording session
(multiple mics: lavs + boom), clean them, and export. Not a parameter-reference
bake-off — a fast, keyboard-driven, batch-capable working tool. Two independent
flows: clean each track individually (Normal), or fuse all mics into one best
vocal track (Merge). Fusion is **additive** — no existing capability is removed.

## Context

- Tracks are 48 kHz / 24-bit PCM. Source set: `Tr1, Tr3, TrA` (mono lavs/boom),
  `TrLR` (stereo mixdown, ~0.82 coherence with Tr3 — a derived monitor mix).
- All tracks are **sample-locked** (cross-correlation lag = 0): recorded on one
  device, sample-synchronous. So time-alignment is unnecessary — the hard part of
  beamforming is free here.
- Iso mics have near-zero pairwise coherence (~0.02): each lav dominantly hears its
  own wearer. Overlapping speech ends up on *different* mics — exploitable.
- Tr1 is hot and clipping (peak 0 dBFS); Tr3/TrA quieter (~-35 dB RMS), cleaner.
- Existing assets reused: `processing.py` (load/save/resample/trim/stats/methods/run),
  5 community rnnoise models in `models/`, DeepFilterNet3 (cached locally, not HF),
  noisereduce. CPU-only on M3 (fast enough at ~2 min clips; MPS skipped — DFN STFT
  ops have spotty MPS support).

## Stack decision

Rebuild the UI as a **local web app: FastAPI backend + vanilla ES-module
frontend + WaveSurfer.js (CDN), no build step.** Chosen over React/Vite (build
toolchain overkill for single-user local tool) and over keeping Gradio (already
hit its walls on trim/zoom/keyboard). WaveSurfer provides waveform render,
zoom/scroll, regions (visual trim), and keyboard hooks out of the box.

## Architecture

```
backend/
  processing.py   reused as-is + new fuse() engine
  server.py       FastAPI: endpoints + serves frontend static files
frontend/
  index.html
  app.js          ES modules
  style.css
  (wavesurfer.js via CDN)
output/<mode>/<track>.wav   24-bit / 48 kHz deliverables
```

### Endpoints

- `POST /session` — body: list of paths (files and/or a folder). Scans, loads each
  track, returns per-track `{name, sr, duration, channels, peaks}` plus a
  session-level `{mergeable: bool, reason: str}`.
- `GET /peaks/{track}` — downsampled waveform peak data for WaveSurfer.
- `GET /audio/{track}` — stream wav for playback (original or rendered).
- `POST /render` — `{track, method, params, trim:[start,end]}` → processes one
  track, saves output, returns `{out_path, stats_before, stats_after}`.
- `POST /render_all` — applies each track's effective settings (global default +
  per-track overrides); per-track isolation (one failure does not abort the batch,
  returns per-track status).
- `POST /merge` — `{params, trim}` → runs fusion engine across mergeable tracks,
  then the chosen denoise on the fused result → one mono master.
- `POST /export` — finalize/copy outputs (location + naming).

## Session & mergeability detection

On import (folder / multi-file / drag-drop — all supported), the backend
determines mergeability:

- **Mergeable** if all tracks share the same sample rate AND near-equal length
  (sample-locked sessions satisfy this).
- **Not mergeable** if sample rates differ or durations differ beyond tolerance
  (e.g. > ~0.5 s). Returns `mergeable:false` with a human reason
  (e.g. "lengths differ 4.2 s — not one session").

UI behavior: incompatible set shows an amber notice and **greys out the Merge
toggle**, but does NOT forbid processing — Normal flow works on any files.

## Two flows (shared param panel)

Shared controls (identical in both flows):
- Method dropdown: `noisereduce`, `deepfilternet`, `rnnoise`,
  `noisereduce__deepfilternet`, `rnnoise__deepfilternet`, `noisereduce__rnnoise`
  (all current methods preserved).
- Params: `nr_stationary`, `nr_prop`, `dfn_atten`, `dfn_mix`, `rnn_model`,
  `rnn_mix` (all current params preserved, including the new DFN wet/dry mix).
- Trim: drag a WaveSurfer region; render processes exactly that region.

**Normal flow:** sidebar lists session tracks. Each track carries its own
method/params, inheriting a session **global default**; any track can **override**.
`Render` processes the focused track; `Render all` processes every track with its
effective settings. Output: one cleaned file per track.

**Merge flow (opt-in, only when mergeable):** fuses all mics into one source via
the fusion engine, then applies the same denoise param panel to the fused result.
Output: one cleaned mono master.

## Fusion engine — `fuse()` (new in processing.py)

Gain-share automatic mic mixer, overlap-safe, exploiting sample-lock:

1. Skip alignment (tracks are sample-locked; lag = 0).
2. Drop derived mixdown tracks from fusion sources when detected (e.g. TrLR's high
   coherence with an iso track marks it as a monitor mix, not an independent mic).
   Configurable; default excludes it.
3. STFT every contributing mic.
4. Per time-frequency cell, estimate each mic's local SNR (cell energy vs that
   mic's own running noise-floor estimate). Convert to a soft weight.
5. Normalize weights across mics per cell and **sum** the weighted cells (NOT
   hard-select). Because speakers separate across mics, simultaneous speech is
   preserved — each voice contributes from its own mic. Idle/noise-only cells get
   suppressed on every mic.
6. Clipping guard: where a mic is railed (Tr1 peak at 0 dBFS), cap its weight so
   clipped garbage does not dominate.
7. Inverse STFT → fused mono track.
8. The user's selected denoise method then runs on the fused track (higher input
   SNR → fewer DFN musical-noise / "pixelation" artifacts on distant voice).

This is the open-source equivalent of Sound Radix Auto-Align Post (alignment —
free here) + a Dugan-style gain-sharing auto-mixer (the per-cell SNR weighting).

## UX & keyboard shortcuts

- Large WaveSurfer waveform: zoom, scroll, drag-region trim. A/B toggle swaps the
  displayed/played buffer between original and cleaned on the same view.
- Status line shows peak / RMS / noise-floor, before → after.
- Shortcuts:
  - `space` play/pause
  - `[` / `]` zoom out / in
  - `←` / `→` seek
  - `i` / `o` set trim in / out
  - `r` render focused track · `⇧R` render all
  - `n` / `p` next / previous track
  - `m` toggle Merge mode (disabled when not mergeable)
  - `a` A/B original↔cleaned
  - `e` export
  - `?` shortcut overlay

## Error handling

- Resample guard retained: force 48 kHz after load (DFN + rnnoise require it;
  browser/editor re-encodes can emit 44.1 k).
- `Render all` isolates per-track failures — reports per-track status, never aborts
  the whole batch on one error.
- Merge disabled (not errored) when session not mergeable.

## Testing

- Smoke-test each endpoint.
- `fuse()` validated on the 3 sample-locked lavs (Tr1/Tr3/TrA): confirm overlap
  preserved, dropout filled (voice on Tr1 cut → carried by Tr3), no comb-filtering
  (sample-locked, so summation is phase-coherent), clipped-mic weight capped.
- Verify all 6 existing methods still run unchanged in Normal flow (no regression).

## Out of scope (YAGNI)

- Per-speaker separated output tracks (one mono master is the Merge deliverable).
- Sub-sample re-alignment (not needed for sample-locked sources).
- React/build tooling, desktop packaging.
- Cloud / external services — fully local.
