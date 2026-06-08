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
        "Type Nepali (Devanagari), choose a voice, and synthesize. Runs **fully offline**.\n"
        "*Quality keeps improving as the model trains.*"
    )
    txt = gr.Textbox(label="Nepali text (नेपाली पाठ)", lines=3, placeholder="नमस्ते, तपाईंलाई कस्तो छ?")
    with gr.Row():
        spk = gr.Dropdown(choices=CHOICES, value=CHOICES[0][1], label=f"Voice ({len(CHOICES)} speakers)")
        speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed (higher = slower)")
        variation = gr.Slider(0.0, 1.0, value=0.667, step=0.01, label="Variation (noise)")
    btn = gr.Button("🔊 Synthesize", variant="primary")
    audio = gr.Audio(label="Output audio", type="filepath", autoplay=True)
    btn.click(synthesize, [txt, spk, speed, variation], audio)
    gr.Examples(
        [["नमस्ते, तपाईंलाई कस्तो छ?"], ["मेरो नाम सूर्यांश हो।"], ["आज काठमाडौंमा मौसम राम्रो छ।"]],
        inputs=txt,
    )
    gr.Markdown(f"<sub>Model: `{os.path.basename(MODEL)}`</sub>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, theme=gr.themes.Soft())
