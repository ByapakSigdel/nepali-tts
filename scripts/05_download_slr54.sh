#!/usr/bin/env bash
# Phase 1, Stage B — download SLR54 (large Nepali ASR, ~17GB, 16 zips + utt_spk_text.tsv).
# Resumable (curl -C -); retries on hiccups; quiet output (-sS) so the log stays readable.
# Does NOT extract — extraction happens later in Stage B preprocessing.
set -u

RAW="$HOME/voicemodel/data/raw/slr54"
mkdir -p "$RAW"
cd "$RAW"
MIRROR="https://openslr.elda.org/resources/54"

for f in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
  echo ">>> downloading asr_nepali_${f}.zip"
  curl -L -C - --retry 8 --retry-delay 5 -sS -o "asr_nepali_${f}.zip" "${MIRROR}/asr_nepali_${f}.zip" \
    || echo "!! asr_nepali_${f}.zip failed (will need re-run)"
done

echo ">>> downloading utt_spk_text.tsv"
curl -L -C - --retry 8 --retry-delay 5 -sS -o utt_spk_text.tsv "${MIRROR}/utt_spk_text.tsv" \
  || echo "!! utt_spk_text.tsv failed"

echo ">>> sizes:"
du -sh "$RAW"
ls -lh "$RAW" | tail -n +1
echo ">>> SLR54 DOWNLOAD STEP DONE (re-run this script to resume any incomplete file)"
