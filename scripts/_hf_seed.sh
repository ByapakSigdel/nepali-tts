#!/usr/bin/env bash
# One-off: push the current local checkpoint + its epoch up to the cloud repo (seeds the baton).
set -uo pipefail
VM="$HOME/voicemodel"
source "$VM/.venv/bin/activate"
export HF_TOKEN="$(cat "$VM/.hf_token")"
export HF_REPO="$(cat "$VM/.hf_repo")"
export DEVICE_ID=laptop
CKPT="$VM/models/ne_stageA/ckpts/last.ckpt"
STATUS="$VM/models/ne_stageA/status.txt"
ep="$(grep -oE '(^| )epoch=[0-9]+' "$STATUS" 2>/dev/null | grep -oE '[0-9]+$' | head -1)"; ep="${ep:--1}"
HS="python /mnt/c/Users/user/Documents/VoiceModel/scripts/hf_sync.py"
echo ">>> seeding cloud with local checkpoint at epoch $ep ($(du -h "$CKPT" | cut -f1))"
$HS push "$CKPT" last.ckpt
$HS push-progress "$ep"
echo ">>> SEED DONE — cloud now holds last.ckpt @ epoch $ep"
