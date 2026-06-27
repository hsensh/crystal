# Building the desktop app (macOS + Windows)

The app = FastAPI server + static frontend, shown in a native window via
`pywebview` (`desktop.py`). The heavy backend (torch / torchaudio /
DeepFilterNet) is **installed on first launch**, not bundled — keeps the
installer small. `ffmpeg` (used by the Leveler and RNNoise) is bundled per-OS.

## Run as a desktop window (dev)
```bash
.venv/bin/pip install pywebview
.venv/bin/python desktop.py
```
First launch installs torch/DFN if missing; the DFN model downloads to a user
cache on first DeepFilterNet render.

## What ships vs installs-on-open
- **Bundled (small):** Python runtime, app code, frontend, fastapi/uvicorn,
  numpy/scipy/soundfile/noisereduce, the rnnoise `models/`, an `ffmpeg` binary.
- **Installed on first open (large):** torch, torchaudio, deepfilternet.
- **Downloaded on first use:** the DeepFilterNet3 model checkpoint (its own cache).

## Packaging (per OS — no cross-build)
Build on a Mac for the Mac app, on Windows for the Windows app (use CI:
GitHub Actions `macos-latest` + `windows-latest`).

### PyInstaller (one folder)
```bash
.venv/bin/pip install pyinstaller pywebview
pyinstaller --noconfirm --windowed --name "Crystal" \
  --add-data "frontend:frontend" \
  --add-data "models:models" \
  --collect-all webview \
  desktop.py
```
- macOS: produces `dist/Crystal.app`. Wrap in a `.dmg` (`create-dmg`).
- Windows: produces `dist/Crystal/Crystal.exe`. Wrap with Inno
  Setup / NSIS for an installer.
- Bundle `ffmpeg`: drop the platform binary in and `--add-binary`, or document a
  one-time download on first run alongside torch.

### First-run install in a frozen app
A frozen build has no `pip` on PATH. Ship a bundled Python's pip and install the
HEAVY packages into a writable app-support dir, added to `sys.path`:
- macOS: `~/Library/Application Support/Crystal/site`
- Windows: `%APPDATA%\Crystal\site`
`resources.py` owns this: it bundles `uv` to install the HEAVY packages into the
site dir (frozen apps have no pip), and the in-app download screen drives it.

## Signing (avoids security warnings)
- **macOS:** Apple Developer account ($99/yr). `codesign` the `.app`, then
  `notarytool submit` + `staple`. Without it: right-click → Open to bypass
  Gatekeeper.
- **Windows:** an Authenticode code-signing cert. Without it: SmartScreen
  "unknown publisher" warning (still installable).

## Realistic effort
- Native window (dev): done (`desktop.py`).
- Unsigned per-OS bundles: ~1–2 days each, mostly PyInstaller + ffmpeg + the
  frozen first-run installer.
- Signed/notarized installers + CI: additional days + the Apple/Windows certs.
