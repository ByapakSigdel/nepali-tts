#!/usr/bin/env python3
"""
Generate Nepali speech with Meta's MMS-TTS (facebook/mms-tts-npi).
This is our NEURAL BASELINE — the quality bar our trained model must beat.

Usage:
  python mms_tts.py [INPUT_TXT] [OUTPUT_DIR]
Reads one Nepali sentence per line, writes one .wav each + a combined file.
"""
import os
import sys
import torch

IN = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/user/Documents/VoiceModel/eval/test_sentences_ne.txt"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "/mnt/c/Users/user/Documents/VoiceModel/eval/mms"
os.makedirs(OUTDIR, exist_ok=True)

from transformers import VitsModel, AutoTokenizer
import numpy as np
import soundfile as sf

MODEL_ID = "facebook/mms-tts-npl"  # Nepali MMS-TTS (ISO code 'npl'); license: cc-by-nc-4.0 (eval/baseline only)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> Loading {MODEL_ID} on {device} (first run downloads ~145MB)...")
model = VitsModel.from_pretrained(MODEL_ID).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
sr = model.config.sampling_rate
is_uroman = getattr(tokenizer, "is_uroman", False)
print(f">>> sampling_rate={sr} Hz, is_uroman={is_uroman}")


import uroman as ur
_uro = ur.Uroman()  # MMS Nepali's vocab is romanized Latin (its is_uroman flag is wrong)


def preprocess(text):
    """Always romanize Devanagari -> Latin, which is what MMS Nepali was trained on."""
    return _uro.romanize_string(text)


lines = [l.strip() for l in open(IN, encoding="utf-8") if l.strip()]
clips = []
for i, line in enumerate(lines, 1):
    text = preprocess(line)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    if inputs.input_ids.shape[1] == 0:
        print(f"  [{i}] {line}  ->  (no recognizable tokens, skipped)")
        continue
    with torch.no_grad():
        wav = model(**inputs).waveform[0].cpu().numpy().astype(np.float32)
    out = os.path.join(OUTDIR, f"mms_ne_{i}.wav")
    sf.write(out, wav, sr)
    clips.append(wav)
    print(f"  [{i}] {line}  ->  [{text}]  ->  {os.path.basename(out)}  ({len(wav)/sr:.1f}s)")

# Stitch all clips with 0.4s gaps into one file for easy listening
gap = np.zeros(int(0.4 * sr), dtype=np.float32)
combined = np.concatenate([seg for w in clips for seg in (w, gap)])
combo = os.path.join(OUTDIR, "mms_ne_all.wav")
sf.write(combo, combined, sr)
print(f">>> Combined: {combo}")
print(">>> MMS BASELINE DONE")
