#!/usr/bin/env bash
# Stop the Stage-B auto-resume loop, the trainer, and the checkpoint guardian.
# Run via:  wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/Documents/VoiceModel/scripts/stop_stageB.sh
# (Patterns live in this file, not the caller's command line, so pkill cannot self-match.)
pkill -f 12_train_stageB.sh 2>/dev/null
sleep 1
pkill -f piper.train 2>/dev/null
pkill -f ckpt_guardian.py 2>/dev/null
sleep 4
# escalate if anything survived
pkill -9 -f piper.train 2>/dev/null
sleep 1
if pgrep -f piper.train >/dev/null; then
  echo "training: STILL RUNNING"
else
  echo "training: stopped"
fi
if pgrep -f 12_train_stageB.sh >/dev/null; then
  echo "resume loop: STILL RUNNING"
else
  echo "resume loop: stopped"
fi
# clean any orphan tmp files from interrupted atomic saves
find ~/voicemodel/models/ne_stageB/cache -name '*.tmp' -delete 2>/dev/null
echo "done"
