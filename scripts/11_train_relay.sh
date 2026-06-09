#!/usr/bin/env bash
# Checkpoint-relay training. Run this INSTEAD of 10_train_autoresume.sh once the Hugging Face
# relay is set up. Used by BOTH machines (home laptop and Colab) — whoever holds the lock trains;
# the other waits. The full-state checkpoint is the baton, passed through a private HF repo.
#
#   Config (env, or files under ~/voicemodel/):
#     HF_TOKEN    write token         (or  ~/voicemodel/.hf_token)
#     HF_REPO     user/nepali-tts-ckpt (or ~/voicemodel/.hf_repo)
#     DEVICE_ID   label for this machine (default: laptop)
#     PUSH_EVERY  seconds between background pushes (default 1800; Colab should use ~1200)
#     FORCE=1     train even if another machine holds the lock (use only if you KNOW it's dead)
#
# Safety: it refuses to pull an OLDER cloud checkpoint over newer local work (compares epochs),
# so no progress is ever lost in either direction.
set -uo pipefail

VM="$HOME/voicemodel"
SCRIPTS="/mnt/c/Users/user/Documents/VoiceModel/scripts"
[ -d "$SCRIPTS" ] || SCRIPTS="$(cd "$(dirname "$0")" && pwd)"   # Colab: scripts sit next to this file
CKPT="$VM/models/ne_stageA/ckpts/last.ckpt"
STATUS="$VM/models/ne_stageA/status.txt"

export HF_TOKEN="${HF_TOKEN:-$(cat "$VM/.hf_token" 2>/dev/null)}"
export HF_REPO="${HF_REPO:-$(cat "$VM/.hf_repo" 2>/dev/null)}"
export DEVICE_ID="${DEVICE_ID:-laptop}"
PUSH_EVERY="${PUSH_EVERY:-1800}"

[ -n "$HF_TOKEN" ] || { echo "!! HF_TOKEN not set (env or ~/voicemodel/.hf_token)"; exit 2; }
[ -n "$HF_REPO" ]  || { echo "!! HF_REPO not set (env or ~/voicemodel/.hf_repo)";  exit 2; }

# shellcheck disable=SC1091
source "$VM/.venv/bin/activate"
python -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub
HS="python $SCRIPTS/hf_sync.py"

epoch_of() { grep -oE '(^| )epoch=[0-9]+' "$STATUS" 2>/dev/null | grep -oE '[0-9]+$' | head -1; }

# 1) take the lock (or bail so we never double-train)
if [ "${FORCE:-0}" != "1" ]; then
  $HS claim || { echo ">>> another machine is training. Try again later, or run with FORCE=1 if it's dead."; exit 3; }
else
  echo ">>> FORCE=1 — claiming lock regardless"; DEVICE_ID="$DEVICE_ID" $HS claim || true
fi

# always release the lock + stop the background pusher when we exit, however we exit
cleanup() { kill "${SYNC_PID:-}" 2>/dev/null; $HS release || true; }
trap cleanup EXIT INT TERM

# 2) sync in the correct direction
remote_ep="$($HS remote-epoch | tail -1)"; remote_ep="${remote_ep:--1}"
local_ep="$(epoch_of)"; local_ep="${local_ep:--1}"
echo ">>> remote epoch=$remote_ep   local epoch=$local_ep"
# Only a PULL must block (we cannot train without the newer checkpoint). Seeding / pushing local
# does NOT block — training already has the correct local ckpt, and the background loop below
# uploads it right away. This keeps the laptop training while the (slow, home-upstream) seed runs.
if [ "$remote_ep" -gt "$local_ep" ]; then
  echo ">>> cloud is AHEAD (epoch $remote_ep > $local_ep) — pulling before training"
  mkdir -p "$(dirname "$CKPT")"
  $HS pull last.ckpt "$CKPT"
elif [ -f "$CKPT" ]; then
  echo ">>> local is the source of truth (epoch $local_ep) — will upload in background, not blocking"
else
  echo ">>> nothing in cloud and no local ckpt — training will fresh-start (vocoder warm-start)"
fi

# 3) background: push immediately (seeds the cloud without blocking), then heartbeat + push on a timer
(
  first=1
  while true; do
    if [ "$first" = 1 ]; then first=0; else sleep "$PUSH_EVERY"; fi
    $HS heartbeat
    if [ -f "$CKPT" ]; then
      ep="$(epoch_of)"; ep="${ep:--1}"
      $HS push "$CKPT" last.ckpt && $HS push-progress "$ep"
    fi
  done
) & SYNC_PID=$!

# 4) train (auto-resumes from local last.ckpt, checkpoints every ~100 steps, stops at 350)
bash "$SCRIPTS/10_train_autoresume.sh"

# 5) final push of whatever we reached, then the trap releases the lock
kill "$SYNC_PID" 2>/dev/null; SYNC_PID=""
if [ -f "$CKPT" ]; then
  ep="$(epoch_of)"; ep="${ep:--1}"
  echo ">>> final push at epoch $ep"
  $HS push "$CKPT" last.ckpt && $HS push-progress "$ep"
fi
echo ">>> relay session complete"
