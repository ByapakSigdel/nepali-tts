#!/usr/bin/env bash
# Upload the few code files Colab needs into the HF repo under code/, so the notebook can fetch them
# with the HF token (no GitHub clone / no making the repo public). Re-run after editing any of them.
set -uo pipefail
VM="$HOME/voicemodel"
source "$VM/.venv/bin/activate"
export HF_TOKEN="$(cat "$VM/.hf_token")"
export HF_REPO="$(cat "$VM/.hf_repo")"
export DEVICE_ID=laptop
HS="python /mnt/c/Users/user/Documents/VoiceModel/scripts/hf_sync.py"
D="/mnt/c/Users/user/Documents/VoiceModel"
$HS push "$D/scripts/hf_sync.py"            code/hf_sync.py
$HS push "$D/scripts/status_writer.py"      code/status_writer.py
$HS push "$D/configs/train_ne_colab.yaml"   code/train_ne_colab.yaml
echo ">>> code uploaded to HF under code/"
