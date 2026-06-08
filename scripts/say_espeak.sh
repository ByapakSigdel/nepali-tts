#!/usr/bin/env bash
# Quick Nepali speech test via espeak-ng (instant, robotic — pipeline sanity check only).
# Reads Nepali text from a UTF-8 file (safer than passing Devanagari on the command line).
#
# Usage:
#   bash say_espeak.sh [INPUT_TXT] [OUTPUT_WAV] [SPEED_WPM]
# Defaults write into the Windows project's eval/ folder so you can play in Windows.
set -euo pipefail

IN="${1:-/mnt/c/Users/user/Documents/VoiceModel/eval/test_sentences_ne.txt}"
OUT="${2:-/mnt/c/Users/user/Documents/VoiceModel/eval/espeak_ne_sample.wav}"
SPEED="${3:-145}"

echo ">>> Synthesizing (espeak-ng, voice=ne, speed=${SPEED} wpm)"
espeak-ng -v ne -s "${SPEED}" -f "${IN}" -w "${OUT}"
echo ">>> Wrote WAV: ${OUT}"

echo ">>> IPA phonemes (eyeball schwa deletion here):"
espeak-ng -v ne -q --ipa -f "${IN}"
