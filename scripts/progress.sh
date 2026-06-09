#!/usr/bin/env bash
# Easy training progress tracker, with ETA.
#   one look:  wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/Documents/VoiceModel/scripts/progress.sh
#   live:      wsl -d Ubuntu-24.04 -- watch -n 15 bash /mnt/c/Users/user/Documents/VoiceModel/scripts/progress.sh
# Set TARGET=<epoch> to change the completion target used for the ETA (default 100).
RUN="${RUN:-ne_stageA}"   # set RUN=ne_stageB to track Stage B
LOG="${LOG:-/mnt/c/Users/user/Documents/VoiceModel/notes/10_autoresume.log}"
CKPT="$HOME/voicemodel/models/$RUN/ckpts/last.ckpt"
STATUS="$HOME/voicemodel/models/$RUN/status.txt"
TARGET="${TARGET:-350}"

epoch_of() { printf '%s' "$1" | grep -oE '(^| )epoch=[0-9]+' | grep -oE '[0-9]+$' | head -1; }
sie_of()   { printf '%s' "$1" | grep -oE 'step_in_epoch=[0-9]+/[0-9]+' | cut -d= -f2; }

# Two reads ~7 s apart to estimate the training rate (fractional epochs / second).
s1=$(cat "$STATUS" 2>/dev/null); t1=$(date +%s)
sleep 7
s2=$(cat "$STATUS" 2>/dev/null); t2=$(date +%s)

ep1=$(epoch_of "$s1"); sie1=$(sie_of "$s1")
ep2=$(epoch_of "$s2"); sie2=$(sie_of "$s2")
loss=$(printf '%s' "$s2" | grep -oE 'loss=[0-9.]+' | cut -d= -f2)
x1=${sie1%/*}; y1=${sie1#*/}; x2=${sie2%/*}; y2=${sie2#*/}

eta=$(awk -v ep1="${ep1:-0}" -v x1="${x1:-0}" -v y1="${y1:-0}" \
          -v ep2="${ep2:-0}" -v x2="${x2:-0}" -v y2="${y2:-0}" \
          -v dt="$((t2 - t1))" -v target="$TARGET" '
function fmt(s,   d, h, m, r) {
  if (s < 0 || s != s) return "--";
  d = int(s/86400); s = s - d*86400; h = int(s/3600); m = int((s - h*3600)/60);
  r = ""; if (d > 0) r = r d"d "; if (h > 0 || d > 0) r = r h"h "; return r m"m";
}
BEGIN {
  if (y1 < 1 || y2 < 1 || dt < 1) { print "estimating... (let it run a few seconds)"; exit }
  ef1 = ep1 + x1/y1; ef2 = ep2 + x2/y2; eps = (ef2 - ef1)/dt;
  if (eps <= 0) { print "estimating... (status not advancing yet)"; exit }
  printf "%.2f it/s, %.1f epochs/hr  |  next epoch ~%s  |  ~%s to epoch %d",
         eps*y2, eps*3600, fmt((1 - x2/y2)/eps), fmt((target - ef2)/eps), target;
}')

ck="(none yet)"; [ -f "$CKPT" ] && ck="$(( $(date +%s) - $(stat -c %Y "$CKPT") ))s ago"
attempts=$(grep -ac ATTEMPT "$LOG" 2>/dev/null); attempts=${attempts:-0}
crashes=$(( attempts > 0 ? attempts - 1 : 0 ))
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
run=$(pgrep -f "[p]iper.train" >/dev/null 2>&1 && echo "🟢 TRAINING" || echo "⚪ not running")

echo "=================== Nepali TTS — Training ==================="
echo "  Status:        $run"
echo "  Epochs done:   ${ep2:-0}"
echo "  This epoch:    ${sie2:-—} batches"
echo "  Loss:          ${loss:-—}"
echo "  Throughput:    $eta"
echo "  Crashes:       $crashes auto-resumed   |   last checkpoint: $ck"
echo "  GPU:           ${gpu:-n/a}"
echo "============================================================"
