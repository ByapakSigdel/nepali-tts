#!/usr/bin/env bash
# Checkpoint-relay training. Run this INSTEAD of 10_train_autoresume.sh to participate in the
# laptop+Colab relay. Whoever holds the lock trains; the other waits. The full-state checkpoint is
# the baton, passed through a private HF repo. Hardened against progress loss (see docs/DEVLOG.md 10.7).
#
#   Config (env, or files under ~/voicemodel/):
#     HF_TOKEN, HF_REPO        (or ~/voicemodel/.hf_token, ~/voicemodel/.hf_repo)
#     DEVICE_ID                label for this machine (default: laptop)
#     PUSH_EVERY               seconds between background checkpoint pushes (default 1800)
#     HEARTBEAT_EVERY          seconds between lock heartbeats (default 120; independent of pushes)
#     LOCK_STALE               seconds before a silent lock is reclaimable (default 900)
#     FORCE=1                  let push-ckpt overwrite a NEWER cloud ckpt (manual override; dangerous)
#
# Safety invariants:
#   * the lock is heartbeated on its OWN fast timer, so a slow 882MB push can't make us look "dead";
#   * we refuse to train if we can't take the lock or can't read the cloud epoch (fail closed);
#   * a mandatory pull is verified before training, so we never train on stale state by accident;
#   * push-ckpt refuses to overwrite a newer cloud checkpoint, so progress can't be lost either way;
#   * if the FINAL push fails, we KEEP the lock (don't hand off) — your epochs stay safe locally.
set -uo pipefail

VM="$HOME/voicemodel"
SCRIPTS="/mnt/c/Users/user/Documents/VoiceModel/scripts"
[ -d "$SCRIPTS" ] || SCRIPTS="$(cd "$(dirname "$0")" && pwd)"   # Colab: scripts sit next to this file
CKPT="$VM/models/ne_stageA/ckpts/last.ckpt"
CONFIG="$VM/models/ne_stageA/config.json"
STATUS="$VM/models/ne_stageA/status.txt"

export HF_TOKEN="${HF_TOKEN:-$(cat "$VM/.hf_token" 2>/dev/null)}"
export HF_REPO="${HF_REPO:-$(cat "$VM/.hf_repo" 2>/dev/null)}"
export DEVICE_ID="${DEVICE_ID:-laptop}"
export LOCK_STALE="${LOCK_STALE:-900}"
PUSH_EVERY="${PUSH_EVERY:-1800}"
HEARTBEAT_EVERY="${HEARTBEAT_EVERY:-120}"

[ -n "$HF_TOKEN" ] || { echo "!! HF_TOKEN not set (env or ~/voicemodel/.hf_token)"; exit 2; }
[ -n "$HF_REPO" ]  || { echo "!! HF_REPO not set (env or ~/voicemodel/.hf_repo)";  exit 2; }

# shellcheck disable=SC1091
source "$VM/.venv/bin/activate"
python -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub
HS="python $SCRIPTS/hf_sync.py"

epoch_of() { grep -oE '(^| )epoch=[0-9]+' "$STATUS" 2>/dev/null | grep -oE '[0-9]+$' | head -1; }

# 1) take the lock (strict — claim itself respects a live holder; FORCE does NOT steal a live lock)
echo ">>> current lock:"; $HS status
$HS claim || { echo ">>> cannot take the lock (busy or unknown). Refusing to train. Try later."; exit 3; }

RELEASE_ON_EXIT=1
cleanup() {
  kill "${HB_PID:-}" "${SYNC_PID:-}" 2>/dev/null
  if [ "$RELEASE_ON_EXIT" = "1" ]; then $HS release || true; fi
}
trap cleanup EXIT INT TERM

# 2) heartbeat on an INDEPENDENT fast timer, so a slow checkpoint upload never starves the lock
( while true; do sleep "$HEARTBEAT_EVERY"; $HS heartbeat >/dev/null 2>&1; done ) & HB_PID=$!

# 3) sync direction — fail closed if we can't trust the cloud epoch
remote_ep="$($HS remote-epoch | tail -1)"
case "$remote_ep" in
  ''|*[!0-9-]*) echo ">>> remote-epoch unreadable ('$remote_ep') — refusing to train (fail closed)."; exit 5;;
esac
local_ep="$(epoch_of)"; local_ep="${local_ep:--1}"
echo ">>> remote epoch=$remote_ep   local epoch=$local_ep"
if [ "$remote_ep" -gt "$local_ep" ]; then
  echo ">>> cloud is AHEAD — pulling checkpoint + config before training (mandatory)"
  mkdir -p "$(dirname "$CKPT")"
  $HS pull last.ckpt "$CKPT" || { echo ">>> PULL FAILED — refusing to train (would overwrite cloud progress)."; exit 4; }
  [ -s "$CKPT" ] || { echo ">>> pulled checkpoint missing/empty — refusing to train."; exit 4; }
  $HS pull config.json "$CONFIG" || echo ">>> warning: config.json pull failed; keeping local config"
elif [ -f "$CKPT" ]; then
  echo ">>> local is the source of truth (epoch $local_ep) — background loop will upload it (guarded)"
else
  echo ">>> nothing in cloud and no local ckpt — training will fresh-start (vocoder warm-start)"
fi

# 4) background: push the checkpoint+config atomically (guarded). First push fires immediately.
( first=1
  while true; do
    if [ "$first" = 1 ]; then first=0; else sleep "$PUSH_EVERY"; fi
    [ -f "$CKPT" ] && $HS push-ckpt "$CKPT" auto --config "$CONFIG"
  done ) & SYNC_PID=$!

# 5) train (auto-resumes from local last.ckpt, checkpoints every ~100 steps, stops at 350)
bash "$SCRIPTS/10_train_autoresume.sh"

# 6) final push — it MUST land before we hand off the lock, or we keep the lock and stay safe locally
kill "$SYNC_PID" 2>/dev/null; SYNC_PID=""
FINAL_OK=0
if [ -f "$CKPT" ]; then
  for attempt in 1 2 3 4 5; do
    if $HS push-ckpt "$CKPT" auto --config "$CONFIG"; then FINAL_OK=1; break; fi
    echo ">>> final push attempt $attempt failed; retrying in 20s"; sleep 20
  done
else
  FINAL_OK=1   # nothing to push
fi
if [ "$FINAL_OK" != "1" ]; then
  echo ">>> !!! FINAL PUSH FAILED. KEEPING the lock so no other machine can overwrite — your latest"
  echo ">>>     epochs are safe locally at $CKPT. Re-run this script once the network is back."
  RELEASE_ON_EXIT=0
  exit 6
fi
echo ">>> relay session complete (epoch $(epoch_of))"
