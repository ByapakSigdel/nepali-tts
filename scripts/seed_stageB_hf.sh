#!/usr/bin/env bash
# Seed the Stage-B checkpoint-relay HF repo (separate from Stage A so locks/epoch
# guards never clash): code files + config.json + the warm-started seed checkpoint.
# Run via:  wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/Documents/VoiceModel/scripts/seed_stageB_hf.sh
set -euo pipefail
source ~/voicemodel/.venv/bin/activate
cd ~/voicemodel

export HF_TOKEN=$(cat ~/voicemodel/.hf_token)
export HF_REPO="byapaksigdel/nepali-tts-ckpt-b"
export DEVICE_ID="laptop"

SYNC=/mnt/c/Users/user/Documents/VoiceModel/scripts/hf_sync.py
WIN=/mnt/c/Users/user/Documents/VoiceModel
RUN=~/voicemodel/models/ne_stageB

echo "== pushing code files =="
python "$SYNC" push "$WIN/scripts/hf_sync.py"                    code/hf_sync.py
python "$SYNC" push "$WIN/scripts/status_writer.py"              code/status_writer.py
python "$SYNC" push "$WIN/configs/train_ne_stageB_colab.yaml"    code/train_ne_stageB_colab.yaml

echo "== pushing config.json =="
python "$SYNC" push "$RUN/config.json" config.json

echo "== pushing seed checkpoint (guarded, epoch auto) =="
python "$SYNC" push-ckpt "$RUN/ckpts/last.ckpt" auto --config "$RUN/config.json"

echo "== done; remote epoch now: =="
python "$SYNC" remote-epoch
