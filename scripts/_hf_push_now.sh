#!/usr/bin/env bash
# One-off: push the current local checkpoint with the new atomic, guarded push-ckpt (epoch read
# from the checkpoint bytes). Refreshes the cloud baton and validates the real upload path.
set -uo pipefail
VM="$HOME/voicemodel"
source "$VM/.venv/bin/activate"
export HF_TOKEN="$(cat "$VM/.hf_token")"
export HF_REPO="$(cat "$VM/.hf_repo")"
export DEVICE_ID=laptop
HS="python /mnt/c/Users/user/Documents/VoiceModel/scripts/hf_sync.py"
$HS push-ckpt "$VM/models/ne_stageA/ckpts/last.ckpt" auto --config "$VM/models/ne_stageA/config.json"
echo "remote epoch now:"; $HS remote-epoch
