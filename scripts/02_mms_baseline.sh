#!/usr/bin/env bash
# Phase 3 (preview) — install baseline deps and generate the MMS Nepali neural sample.
set -euo pipefail
VM="$HOME/voicemodel"
# shellcheck disable=SC1091
source "$VM/.venv/bin/activate"

echo ">>> Installing baseline deps (numpy, soundfile, transformers, uroman)..."
pip install -q numpy soundfile "transformers>=4.44" uroman

echo ">>> Generating MMS Nepali baseline..."
python /mnt/c/Users/user/Documents/VoiceModel/scripts/mms_tts.py
