#!/usr/bin/env bash
# Phase 1, Stage A — download the TTS-grade Nepali datasets (SLR43 + SLR143) and extract.
# Saved to WSL-native fs for fast I/O. Resumable (curl -C -).
set -euo pipefail

RAW="$HOME/voicemodel/data/raw"
mkdir -p "$RAW"
cd "$RAW"

MIRROR="https://openslr.elda.org/resources"

echo ">>> [1/4] SLR43 — multi-speaker female TTS (~800MB, CC BY-SA 4.0)"
curl -L -C - --retry 3 --retry-delay 5 -o ne_np_female.zip "$MIRROR/43/ne_np_female.zip"

echo ">>> [2/4] SLR143 — male+female TTS (~165MB, CC BY-NC-SA 4.0)"
curl -L -C - --retry 3 --retry-delay 5 -o slr143_male_female.tgz "$MIRROR/143/male-female-data.tgz"

echo ">>> [3/4] Extracting (python zipfile + tar; no unzip needed)"
mkdir -p slr43 slr143
python3 -m zipfile -e ne_np_female.zip slr43/
tar -xzf slr143_male_female.tgz -C slr143

echo ">>> [4/4] Layout + sizes"
echo "--- SLR43 top level ---"; find slr43 -maxdepth 2 | head -30
echo "--- SLR143 top level ---"; find slr143 -maxdepth 3 | head -30
echo "--- disk usage ---"; du -sh slr43 slr143 2>/dev/null
echo ">>> DOWNLOAD + EXTRACT COMPLETE"
