"""Gradio UI: tune denoise params per method, A/B compare, save outputs.

Trim is visual: the Original player is an editable waveform — drag the handles
to select the region you want. "Zoom window" reloads only a sub-range into the
player so the waveform stretches for fine selection. Render processes exactly
what's currently in the Original player.

Run:  .venv/bin/python app.py   (then open the printed local URL)
Outputs saved to output/<method>/<track>.wav (24-bit / 48 kHz).
"""
import glob
import os

import gradio as gr

import processing as P

ROOT = os.path.dirname(__file__)
FILES_DIR = os.path.join(ROOT, "files")
OUT_DIR = os.path.join(ROOT, "output")

TRACKS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FILES_DIR, "*.WAV")))


def fmt_stats(label, s):
    return (f"**{label}** — peak `{s['peak_dbfs']:.1f}` dBFS · "
            f"RMS `{s['rms_dbfs']:.1f}` dBFS · "
            f"noise floor `{s['noise_floor_dbfs']:.1f}` dBFS")


def load_window(track, zoom_start, zoom_end):
    """Load (a window of) the track into the editable player for visual trim."""
    if not track:
        return None, ""
    path = os.path.join(FILES_DIR, track)
    audio, sr = P.load(path)
    sliced = P.trim(audio, sr, zoom_start, zoom_end)
    dur = sliced.shape[1] / sr
    lbl = "Original" if (float(zoom_start) <= 0 and float(zoom_end) <= 0) else \
          f"Original — zoom {float(zoom_start):g}-{float(zoom_end):g}s"
    return P.write_temp(sliced, sr), fmt_stats(f"{lbl} ({dur:.1f}s)", P.stats(sliced))


def process(orig_clip, track, method,
            nr_stationary, nr_prop, dfn_atten, dfn_mix, rnn_model_label, rnn_mix):
    if not orig_clip:
        raise gr.Error("No audio in the Original player. Pick a track first.")
    audio, sr = P.load(orig_clip)  # exactly what user trimmed/zoomed in the player
    audio, sr = P.resample(audio, sr)  # Gradio editor may emit 44.1k; force 48k
    params = dict(
        stationary=nr_stationary,
        prop_decrease=nr_prop,
        atten_lim_db=dfn_atten,
        dfn_mix=dfn_mix,
        rnn_model=P.RNN_MODELS[rnn_model_label],
        rnn_mix=rnn_mix,
    )
    out = P.run(method, audio, sr, **params)
    dur = audio.shape[1] / sr
    base = (track or "clip").lower()
    stem = os.path.splitext(base)[0]
    # tag with duration when it's a trimmed clip (shorter than full 121s track)
    name = base if dur > 120 else f"{stem}_{dur:.1f}s.wav"
    out_path = os.path.join(OUT_DIR, method, name)
    P.save(out, sr, out_path)
    info = fmt_stats(f"Cleaned ({method})", P.stats(out))
    return out_path, f"{info}\n\nSaved → `output/{method}/{name}`"


WF = gr.WaveformOptions(waveform_color="#3b82f6", trim_region_color="#f59e0b")

with gr.Blocks(title="Film dialogue denoise — compare") as demo:
    gr.Markdown("# Film dialogue denoise — method/param bake-off\n"
                "**Trim visually**: drag the handles on the Original waveform to select a region. "
                "Use **Zoom window** to stretch a sub-range for fine selection. "
                "**Render** processes exactly what's in the Original player. "
                "Saved to `output/<method>/<track>.wav`.")

    with gr.Row():
        track = gr.Dropdown(TRACKS, label="Track", value=TRACKS[0] if TRACKS else None)
        method = gr.Dropdown(list(P.METHODS), label="Method", value="deepfilternet")

    with gr.Row():
        zoom_start = gr.Number(0, label="Zoom start (s)", minimum=0, precision=2)
        zoom_end = gr.Number(0, label="Zoom end (s, 0 = full)", minimum=0, precision=2)
        zoom_btn = gr.Button("Zoom window")
        reset_btn = gr.Button("Full view")

    with gr.Accordion("Parameters (each method uses the relevant ones)", open=True):
        with gr.Row():
            nr_stationary = gr.Checkbox(True, label="noisereduce: stationary")
            nr_prop = gr.Slider(0, 1, 0.8, step=0.05, label="noisereduce: strength (prop_decrease)")
        with gr.Row():
            dfn_atten = gr.Slider(0, 60, 0, step=1,
                                  label="DeepFilterNet: max attenuation dB (0 = unlimited / most aggressive; "
                                        "raise to ~12-18 to stop distant voice getting crushed/pixelated)")
            dfn_mix = gr.Slider(0, 1, 1.0, step=0.05,
                                label="DeepFilterNet: wet/dry mix (lower re-adds original to mask artifacts "
                                      "on distant/quiet voice)")
        with gr.Row():
            rnn_model = gr.Dropdown(list(P.RNN_MODELS), label="rnnoise: model",
                                    value=list(P.RNN_MODELS)[0])
            rnn_mix = gr.Slider(0, 1, 1.0, step=0.05,
                                label="rnnoise: wet/dry mix (lower keeps more original / transients)")

    render = gr.Button("Render", variant="primary")

    with gr.Row():
        with gr.Column():
            orig_audio = gr.Audio(label="Original — drag handles to trim", type="filepath",
                                  editable=True, interactive=True, waveform_options=WF)
            orig_stats = gr.Markdown()
        with gr.Column():
            out_audio = gr.Audio(label="Cleaned", type="filepath", waveform_options=WF)
            out_stats = gr.Markdown()

    track.change(load_window, [track, zoom_start, zoom_end], [orig_audio, orig_stats])
    demo.load(load_window, [track, zoom_start, zoom_end], [orig_audio, orig_stats])
    zoom_btn.click(load_window, [track, zoom_start, zoom_end], [orig_audio, orig_stats])
    reset_btn.click(lambda t: load_window(t, 0, 0), track, [orig_audio, orig_stats])
    render.click(
        process,
        [orig_audio, track, method, nr_stationary, nr_prop, dfn_atten, dfn_mix, rnn_model, rnn_mix],
        [out_audio, out_stats],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
