#!/usr/bin/env bash
# Unified live progress dashboard — renders bars for whatever is active:
# tar packaging, HF upload, HF download, and Stage-B training.
# Watch it live:
#   wsl -d Ubuntu-24.04 -- bash -lc 'watch -n 5 bash /mnt/c/Users/user/Documents/VoiceModel/scripts/progress_all.sh'
NOTES=/mnt/c/Users/user/Documents/VoiceModel/notes
VM="$HOME/voicemodel"

bar() {  # bar <pct>
  local pct=${1:-0} w=34 i fill out=""
  (( pct<0 )) && pct=0; (( pct>100 )) && pct=100
  fill=$(( pct*w/100 ))
  for ((i=0;i<w;i++)); do (( i<fill )) && out+="#" || out+="."; done
  printf "[%s] %3d%%" "$out" "$pct"
}
gb() { awk "BEGIN{printf \"%.1f\", $1/1073741824}"; }

echo "============ Nepali TTS — live progress  $(date +%H:%M:%S) ============"

# ---- TAR packaging ----
TAR="$VM/data/processed_b.tar.gz"
if pgrep -f "tar -C .*processed_b" >/dev/null 2>&1; then
  sz=$(stat -c %s "$TAR" 2>/dev/null || echo 0); est=$((16*1024*1024*1024))
  printf "TAR  pack   %s  %sG/~16G\n" "$(bar $((sz*100/est)))" "$(gb $sz)"
elif grep -q TAR-DONE "$NOTES/tar_stageB.log" 2>/dev/null; then
  printf "TAR  done   %s  (%s)\n" "$(bar 100)" "$(du -h "$TAR" 2>/dev/null | cut -f1)"
fi

# ---- HF upload (parse huggingface_hub tqdm; \r-separated, two-pipe bars) ----
ULOG="$NOTES/upload_stageB.log"
if [ -f "$ULOG" ]; then
  if grep -aq "UPLOAD OK" "$ULOG"; then
    printf "UP   done   %s  data on HF\n" "$(bar 100)"
  else
    line=$(tr '\r' '\n' < "$ULOG" | grep -a "Processing Files" | tail -1)
    pct=$(echo "$line" | grep -oE "[0-9]+%" | head -1 | tr -d '%')
    detail=$(echo "$line" | grep -oE "[0-9.]+[KMG]B / [0-9.]+[KMG]B, [0-9.]+[KMG]B/s")
    if [ -n "${pct:-}" ]; then
      printf "UP   send   %s  %s\n" "$(bar ${pct})" "${detail:-uploading}"
    else
      printf "UP   prep   %s  hashing file...\n" "$(bar 0)"
    fi
  fi
fi

# ---- Training ----
ST="$VM/models/ne_stageB/status.txt"
if pgrep -f "piper.train" >/dev/null 2>&1 && [ -f "$ST" ]; then
  read_st=$(cat "$ST" 2>/dev/null)
  ep=$(echo "$read_st"   | grep -oE "epoch=[0-9]+"            | cut -d= -f2)
  sie=$(echo "$read_st"  | grep -oE "step_in_epoch=[0-9]+/[0-9]+" | cut -d= -f2)
  loss=$(echo "$read_st" | grep -oE "loss=[0-9.]+"           | cut -d= -f2)
  cur=${sie%/*}; tot=${sie#*/}
  [ -n "${tot:-}" ] && [ "$tot" -gt 0 ] 2>/dev/null \
    && printf "TRN  e%s    %s  step %s  loss %s\n" "${ep:-?}" "$(bar $((cur*100/tot)))" "${sie}" "${loss:-?}"
fi

# ---- Caching (before training, when prepare_data runs) ----
CACHE="$VM/models/ne_stageB/cache"
if pgrep -f "piper.train" >/dev/null 2>&1 && [ ! -f "$ST" ] && [ -d "$CACHE" ]; then
  c=$(ls "$CACHE" 2>/dev/null | grep -c "audio.pt")
  printf "CACHE       %s  %s/128009 clips\n" "$(bar $((c*100/128009)))" "$c"
fi

echo "======================================================================"
