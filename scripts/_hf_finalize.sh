#!/usr/bin/env bash
# Push config.json (needed by Colab) and report cloud state. Safe to re-run.
set -uo pipefail
VM="$HOME/voicemodel"
source "$VM/.venv/bin/activate"
export HF_TOKEN="$(cat "$VM/.hf_token")"
export HF_REPO="$(cat "$VM/.hf_repo")"
export DEVICE_ID=laptop
HS="python /mnt/c/Users/user/Documents/VoiceModel/scripts/hf_sync.py"
echo "== push config.json =="
$HS push "$VM/models/ne_stageA/config.json" config.json
echo "== remote epoch =="
$HS remote-epoch
echo "== lock status =="
$HS status
