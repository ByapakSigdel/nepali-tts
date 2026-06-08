#!/usr/bin/env python3
"""
Offline Nepali Text-to-Speech frontend (Gradio).
Type Nepali (Devanagari) -> choose a voice -> get audio. Runs locally, no internet needed.

Model is the exported ONNX: NE_TTS_MODEL env var, else ~/voicemodel/frontend/model.onnx,
else falls back to ~/voicemodel/_sample/ne.onnx. Synthesis shells out to the proven `python -m piper`.
"""
import os
import json
import tempfile
import subprocess

import gradio as gr

from translit import to_devanagari  # romanized Nepali / code-mixed English -> Devanagari (offline, no torch)

# ---- locate model ----
MODEL = os.environ.get("NE_TTS_MODEL") or os.path.expanduser("~/voicemodel/frontend/model.onnx")
if not os.path.exists(MODEL):
    MODEL = os.path.expanduser("~/voicemodel/_sample/ne.onnx")
CONFIG = MODEL + ".json"

with open(CONFIG, encoding="utf-8") as f:
    cfg = json.load(f)
num_speakers = int(cfg.get("num_speakers", 1))
smap = cfg.get("speaker_id_map", {}) or {}
if smap:
    CHOICES = [(name, int(sid)) for name, sid in sorted(smap.items(), key=lambda kv: kv[1])]
else:
    CHOICES = [(f"Speaker {i}", i) for i in range(num_speakers)]


def synthesize(text, speaker_id, speed, variation):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter some Nepali text.")
    out = tempfile.mktemp(suffix=".wav")
    cmd = [
        "python", "-m", "piper", "-m", MODEL, "-s", str(int(speaker_id)),
        "--length-scale", str(speed), "--noise-scale", str(variation), "-f", out,
    ]
    r = subprocess.run(cmd, input=(text + "\n").encode("utf-8"), stderr=subprocess.PIPE)
    if r.returncode != 0 or not os.path.exists(out):
        raise gr.Error("Synthesis failed: " + r.stderr.decode("utf-8", "ignore")[-300:])
    return out


with gr.Blocks(title="Nepali TTS") as demo:
    gr.Markdown(
        "# 🗣️ Nepali Text-to-Speech\n"
        "Type Nepali in **plain English letters** (e.g. *kasto cha tapaiko aaile*) — or Devanagari. "
        "It auto-converts to Devanagari below; **edit that box** if anything's off, then synthesize. "
        "Runs **fully offline**. *Quality keeps improving as the model trains.*"
    )
    inp = gr.Textbox(
        label="Type here — romanized Nepali (Latin) or Devanagari",
        lines=2, placeholder="kasto cha tapaiko aaile   /   नमस्ते, तपाईंलाई कस्तो छ?",
    )
    deva = gr.Textbox(
        label="Devanagari (auto — editable; fix before synthesizing)",
        lines=2, interactive=True,
    )
    inp.change(to_devanagari, inputs=inp, outputs=deva)
    with gr.Row():
        spk = gr.Dropdown(choices=CHOICES, value=CHOICES[0][1], label=f"Voice ({len(CHOICES)} speakers)")
        speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed (higher = slower)")
        variation = gr.Slider(0.0, 1.0, value=0.667, step=0.01, label="Variation (noise)")
    btn = gr.Button("🔊 Synthesize", variant="primary")
    audio = gr.Audio(label="Output audio", type="filepath", autoplay=True)
    btn.click(synthesize, inputs=[deva, spk, speed, variation], outputs=audio)
    gr.Examples(
        [["kasto cha tapaiko aaile"],
         ["hello sanchai hunuhuncha tapai"],
         ["mero naam suryansh ho"],
         ["नमस्ते, तपाईंलाई कस्तो छ?"]],
        inputs=inp,
    )
    gr.Markdown(f"<sub>Model: `{os.path.basename(MODEL)}` · romanized→Devanagari via indic-transliteration (offline)</sub>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, theme=gr.themes.Soft())
