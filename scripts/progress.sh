#!/usr/bin/env bash
# Easy training progress tracker.
#   One look:   wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/Documents/VoiceModel/scripts/progress.sh
#   Live (auto-refresh every 10s):
#     wsl -d Ubuntu-24.04 -- watch -n 10 bash /mnt/c/Users/user/Documents/VoiceModel/scripts/progress.sh
LOG="/mnt/c/Users/user/Documents/VoiceModel/notes/10_autoresume.log"
CKPT="$HOME/voicemodel/models/ne_stageA/ckpts/last.ckpt"

line=$(grep -aoE "Epoch [0-9]+/[0-9-]+ +[^0-9]*[0-9]+/[0-9]+ +[0-9:]+ +. +[0-9:]+ +[0-9.]+it/s" "$LOG" 2>/dev/null | tail -1)
epoch_cur=$(printf '%s' "$line" | grep -oE "Epoch [0-9]+" | grep -oE "[0-9]+$")
within=$(printf '%s' "$line"  | grep -oE "[0-9]+/[0-9]+ +[0-9:]+ +. +[0-9:]+" | head -1)
speed=$(printf '%s' "$line"   | grep -oE "[0-9.]+it/s" | tail -1)

attempts=$(grep -c "ATTEMPT" "$LOG" 2>/dev/null); attempts=${attempts:-0}
crashes=$(( attempts > 0 ? attempts - 1 : 0 ))

ckpt_time="(none yet)"; ckpt_ago=""
if [ -f "$CKPT" ]; then
  ckpt_time=$(stat -c %y "$CKPT" 2>/dev/null | cut -d. -f1)
  ckpt_ago="($(( ($(date +%s) - $(stat -c %Y "$CKPT")) / 60 )) min ago)"
fi
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
running=$(pgrep -f "[p]iper.train" >/dev/null && echo "🟢 TRAINING" || echo "⚪ not running")

echo "=================== Nepali TTS — Training ==================="
echo "  Status:          $running"
echo "  Epoch (current): ${epoch_cur:-0}        <- this many epochs done"
echo "  This epoch:      ${within:-— (starting up)}"
echo "  Speed:           ${speed:-—}"
echo "  Crashes handled: $crashes  (auto-resumed, no progress lost)"
echo "  Last checkpoint: $ckpt_time $ckpt_ago"
echo "  GPU now:         ${gpu:-n/a}"
echo "============================================================"
