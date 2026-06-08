#!/usr/bin/env bash
# Phase 0 — system-level setup for Nepali TTS training (run inside WSL Ubuntu-24.04).
# This is the ONLY step that requires sudo. Everything else installs into the venv.
set -euo pipefail

echo ">>> Updating apt and installing system packages..."
sudo apt update
sudo apt install -y \
  python3.12-venv python3-pip python3-dev build-essential \
  ffmpeg espeak-ng espeak-ng-data libespeak-ng-dev libsndfile1 sox git

echo ">>> Verifying tools..."
ffmpeg  -version | head -1
espeak-ng --version
echo ">>> Checking Nepali (ne) voice in espeak-ng..."
espeak-ng --voices=ne || echo "WARN: no 'ne' voice listed — will investigate phonemizer fallback"
echo ">>> Sample Nepali phonemization (नमस्ते):"
espeak-ng -v ne -q --ipa "नमस्ते" || true

echo ">>> Done. Next: create venv + install PyTorch (no sudo needed)."
