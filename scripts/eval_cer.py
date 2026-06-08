#!/usr/bin/env python3
"""
Objective intelligibility eval: ASR round-trip Character Error Rate (CER).
Synthesize each test sentence with the TTS model, transcribe it back with a Nepali ASR
(Whisper), and compare to the input text. Lower CER/WER = more intelligible/accurate.

NOTE: meaningful only once the model is sufficiently trained — very early checkpoints
produce ~100% CER. First run downloads Whisper-large-v3 (~3 GB).

Deps:  pip install faster-whisper jiwer
Usage: python eval_cer.py [ONNX_MODEL] [TEST_TXT] [SPEAKER_ID] [WHISPER_SIZE]
"""
import os
import sys
import subprocess
import tempfile
import unicodedata

ONNX = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/voicemodel/_sample/ne.onnx")
TXT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/c/Users/user/Documents/VoiceModel/eval/test_sentences_ne.txt"
SPK = sys.argv[3] if len(sys.argv) > 3 else "0"
WHISPER = sys.argv[4] if len(sys.argv) > 4 else "large-v3"

import jiwer
from faster_whisper import WhisperModel


def synth(text, wav):
    subprocess.run(
        ["python", "-m", "piper", "-m", ONNX, "-s", SPK, "-f", wav],
        input=(text + "\n").encode("utf-8"), check=True, stderr=subprocess.DEVNULL,
    )


def norm(s):
    """NFC-normalize and strip punctuation/whitespace so CER reflects pronunciation, not formatting."""
    s = unicodedata.normalize("NFC", s).strip()
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith(("P", "Z")))


def main():
    lines = [l.strip() for l in open(TXT, encoding="utf-8") if l.strip()]
    print(f">>> {len(lines)} sentences | model={ONNX} | speaker={SPK} | ASR=Whisper-{WHISPER}")
    asr = WhisperModel(WHISPER, device="cpu", compute_type="int8")

    refs, hyps = [], []
    for i, line in enumerate(lines, 1):
        wav = tempfile.mktemp(suffix=".wav")
        synth(line, wav)
        segments, _ = asr.transcribe(wav, language="ne")
        hyp = "".join(s.text for s in segments)
        refs.append(norm(line))
        hyps.append(norm(hyp))
        print(f"  [{i}] REF: {line}")
        print(f"      ASR: {hyp.strip()}")

    cer = jiwer.cer(refs, hyps) * 100
    wer = jiwer.wer(refs, hyps) * 100
    print(f"\n=== CER = {cer:.1f}%   WER = {wer:.1f}%   (lower = better; over {len(refs)} sentences) ===")


if __name__ == "__main__":
    main()
