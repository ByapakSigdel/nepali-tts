#!/usr/bin/env bash
# Generate Nepali audio from the CURRENT training checkpoint (safe to run while training continues).
# Copies the checkpoint (race-safe), exports to ONNX, synthesizes on CPU across a few speakers.
set -uo pipefail
# shellcheck disable=SC1091
source ~/voicemodel/.venv/bin/activate

RUN="$HOME/voicemodel/models/ne_stageA"
CFG="$RUN/config.json"
TMP="$HOME/voicemodel/_sample"; mkdir -p "$TMP"
OUT="/mnt/c/Users/user/Documents/VoiceModel/eval/ne_model"; mkdir -p "$OUT"
TXT="/mnt/c/Users/user/Documents/VoiceModel/eval/test_sentences_ne.txt"

echo ">>> [1/3] copy current checkpoint (atomic, race-safe with training)"
cp "$RUN/ckpts/last.ckpt" "$TMP/cur.ckpt"
ls -la "$TMP/cur.ckpt"

echo ">>> [2/3] export to ONNX (CPU)..."
python -m piper.train.export_onnx --checkpoint "$TMP/cur.ckpt" --output-file "$TMP/ne.onnx"
cp "$CFG" "$TMP/ne.onnx.json"   # piper inference looks for <model>.onnx.json next to the model
ls -la "$TMP/ne.onnx"

echo ">>> [3/3] synthesize (CPU)..."
mapfile -t LINES < "$TXT"
synth() {  # speaker_id  text  outfile
  printf '%s\n' "$2" | python -m piper -m "$TMP/ne.onnx" -s "$1" -f "$3" 2>/dev/null \
    && echo "   ok -> $(basename "$3")" || echo "   FAILED spk $1"
}
synth 0  "${LINES[0]}" "$OUT/spk0_s1.wav"
synth 0  "${LINES[1]}" "$OUT/spk0_s2.wav"
synth 0  "${LINES[2]}" "$OUT/spk0_s3.wav"
synth 7  "${LINES[0]}" "$OUT/spk7_s1.wav"
synth 14 "${LINES[0]}" "$OUT/spk14_s1.wav"

echo ">>> DONE. Files:"
ls -la "$OUT"/*.wav 2>/dev/null
