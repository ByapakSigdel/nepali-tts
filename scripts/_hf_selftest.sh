#!/usr/bin/env bash
# Quick validation of the relay sync engine (lock mechanics only — no big uploads).
set -uo pipefail
VM="$HOME/voicemodel"
source "$VM/.venv/bin/activate"
export HF_TOKEN="$(cat "$VM/.hf_token")"
export HF_REPO="$(cat "$VM/.hf_repo")"
export DEVICE_ID=selftest
HS="python /mnt/c/Users/user/Documents/VoiceModel/scripts/hf_sync.py"
echo "== status (expect FREE) =="; $HS status
echo "== claim =="; $HS claim
echo "== status (expect held by selftest) =="; $HS status
echo "== remote-epoch (expect -1) =="; $HS remote-epoch
echo "== release =="; $HS release
echo "== status (expect FREE) =="; $HS status
echo "ALL OK"
