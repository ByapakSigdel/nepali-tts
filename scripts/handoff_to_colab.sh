#!/usr/bin/env bash
# Hand the Stage-B baton from the laptop to Colab:
#   1) stop local training + guardian cleanly
#   2) push the latest VALID checkpoint to the relay repo (so Colab resumes from it)
#   3) confirm the lock is free + report the remote epoch
# Run:  wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/Documents/VoiceModel/scripts/handoff_to_colab.sh
set -uo pipefail
# shellcheck disable=SC1091
source ~/voicemodel/.venv/bin/activate

export HF_TOKEN="$(cat ~/voicemodel/.hf_token)"
export HF_REPO="byapaksigdel/nepali-tts-ckpt-b"   # Stage-B relay repo (separate from Stage A)
export DEVICE_ID="laptop"
SYNC=/mnt/c/Users/user/Documents/VoiceModel/scripts/hf_sync.py
PICK=/mnt/c/Users/user/Documents/VoiceModel/scripts/pick_ckpt.py
RUN="$HOME/voicemodel/models/ne_stageB"

echo ">>> [1/3] stopping local training + guardian..."
pkill -f 12_train_stageB.sh 2>/dev/null; sleep 1
pkill -f piper.train 2>/dev/null; pkill -f ckpt_guardian.py 2>/dev/null; sleep 4
pkill -9 -f piper.train 2>/dev/null; sleep 1
if pgrep -f piper.train >/dev/null; then echo "    WARNING: trainer still alive"; else echo "    stopped."; fi
find "$RUN/cache" -name '*.tmp' -delete 2>/dev/null

echo ">>> [2/3] selecting + pushing the latest VALID checkpoint..."
CK="$(python "$PICK")"
if [ -z "$CK" ]; then echo "    !! no valid checkpoint found — aborting (nothing pushed)"; exit 1; fi
echo "    pushing: $CK"
python "$SYNC" push-ckpt "$CK" auto --config "$RUN/config.json" || { echo "    !! push failed"; exit 1; }

echo ">>> [3/3] lock status + remote epoch:"
python "$SYNC" status
printf "    remote epoch is now: "; python "$SYNC" remote-epoch
echo ">>> HANDOFF COMPLETE — Colab can Run all and it will resume from the above epoch."
