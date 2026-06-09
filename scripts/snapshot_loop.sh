#!/usr/bin/env bash
# Accumulate generator snapshots for the end-of-run model-soup / EMA export. Non-disruptive:
# reads last.ckpt and extracts model_g only (~120 MB each). Snapshots once per SNAP_EVERY epochs.
#   launch in background:  bash scripts/snapshot_loop.sh
set -uo pipefail
VM="$HOME/voicemodel"
STATUS="$VM/models/ne_stageA/status.txt"
CKPT="$VM/models/ne_stageA/ckpts/last.ckpt"
SNAP="/mnt/c/Users/user/Documents/VoiceModel/scripts/snapshot_generator.py"
SNAP_EVERY="${SNAP_EVERY:-15}"
source "$VM/.venv/bin/activate"
epoch_of() { grep -oE '(^| )epoch=[0-9]+' "$STATUS" 2>/dev/null | grep -oE '[0-9]+$' | head -1; }

echo ">>> snapshot loop started (every $SNAP_EVERY epochs)"
last_snap=-1
while true; do
  ep="$(epoch_of)"; ep="${ep:--1}"
  if [ "$ep" -ge 0 ] && [ "$ep" != "$last_snap" ] && [ $(( ep % SNAP_EVERY )) -eq 0 ] && [ -f "$CKPT" ]; then
    python "$SNAP" "$CKPT" && last_snap="$ep"
  fi
  sleep 120
done
