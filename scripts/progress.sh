#!/usr/bin/env bash
# Easy training progress tracker.
#   One look:  wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/Documents/VoiceModel/scripts/progress.sh
#   Live:      wsl -d Ubuntu-24.04 -- watch -n 10 bash /mnt/c/Users/user/Documents/VoiceModel/scripts/progress.sh
LOG="/mnt/c/Users/user/Documents/VoiceModel/notes/10_autoresume.log"
CKPT="$HOME/voicemodel/models/ne_stageA/ckpts/last.ckpt"
STATUS="$HOME/voicemodel/models/ne_stageA/status.txt"
TOTAL_EPOCHS=2000

# --- live numbers from the status file the trainer writes ---
epoch="0"; gstep="0"; sie="(starting up...)"; loss="—"; sage="never"
if [ -f "$STATUS" ]; then
  s=$(cat "$STATUS")
  epoch=$(printf '%s' "$s" | grep -oE '(^| )epoch=[0-9]+' | grep -oE '[0-9]+$' | head -1)
  gstep=$(printf '%s' "$s" | grep -oE 'global_step=[0-9]+' | cut -d= -f2)
  sie=$(printf '%s'   "$s" | grep -oE 'step_in_epoch=[0-9]+/[0-9]+' | cut -d= -f2)
  l=$(printf '%s'     "$s" | grep -oE 'loss=[0-9.]+' | cut -d= -f2); [ -n "$l" ] && loss="$l"
  ts=$(printf '%s'    "$s" | grep -oE 'ts=[0-9]+' | cut -d= -f2)
  [ -n "$ts" ] && sage="$(( $(date +%s) - ts ))s ago"
fi

ck="(none yet)"
if [ -f "$CKPT" ]; then
  ck="$(stat -c %y "$CKPT" | cut -d. -f1)  ($(( ($(date +%s) - $(stat -c %Y "$CKPT")) ))s ago)"
fi
attempts=$(grep -ac "ATTEMPT" "$LOG" 2>/dev/null); attempts=${attempts:-0}
crashes=$(( attempts > 0 ? attempts - 1 : 0 ))
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
run=$(pgrep -f "[p]iper.train" >/dev/null 2>&1 && echo "🟢 TRAINING" || echo "⚪ not running")

echo "=================== Nepali TTS — Training ==================="
echo "  Status:           $run"
echo "  Epochs done:      ${epoch:-0} / $TOTAL_EPOCHS"
echo "  Step in epoch:    ${sie:-—}     (updated $sage)"
echo "  Total steps:      ${gstep:-0}"
echo "  Loss:             $loss        (lower = better; very rough early)"
echo "  Crashes handled:  $crashes  (auto-resumed)"
echo "  Last checkpoint:  $ck"
echo "  GPU:              ${gpu:-n/a}"
echo "============================================================"
