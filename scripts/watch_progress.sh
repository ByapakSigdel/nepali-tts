#!/usr/bin/env bash
# Live, auto-refreshing training monitor. Defaults to Stage B. Ctrl+C exits the VIEWER only —
# training keeps running in the background.
#   Stage B:  wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/Documents/VoiceModel/scripts/watch_progress.sh
#   Stage A:  RUN=ne_stageA  ... bash watch_progress.sh
#   tweak:    INTERVAL=10 TARGET=100  ... bash watch_progress.sh
RUN="${RUN:-ne_stageB}"
INTERVAL="${INTERVAL:-15}"
case "$RUN" in
  ne_stageB) LOG="${LOG:-/mnt/c/Users/user/Documents/VoiceModel/notes/12_stageB.log}"; TARGET="${TARGET:-100}" ;;
  *)         LOG="${LOG:-/mnt/c/Users/user/Documents/VoiceModel/notes/10_autoresume.log}"; TARGET="${TARGET:-350}" ;;
esac
export RUN LOG TARGET
exec watch -n "$INTERVAL" -t bash /mnt/c/Users/user/Documents/VoiceModel/scripts/progress.sh
