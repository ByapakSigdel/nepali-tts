#!/usr/bin/env bash
# Phase 4 — auto-resuming trainer. Works around the intermittent Blackwell/WSL CUDA fault by
# checkpointing every 100 steps and restarting from last.ckpt whenever training crashes.
set -uo pipefail
# shellcheck disable=SC1091
source ~/voicemodel/.venv/bin/activate
cd ~/voicemodel/piper1-gpl

export PYTORCH_JIT=0           # Blackwell: avoid TorchScript fused-op crash
# CUDA_LAUNCH_BLOCKING=1 stabilized the fault but was ~50x too slow (~1 step/min) — disabled.
# Strategy: run FAST (accept crashes), checkpoint every 25 steps, auto-resume from last.ckpt.

VM="$HOME/voicemodel"
DATA="$VM/data/processed"
RUN="$VM/models/ne_stageA"
CKPT="$VM/checkpoints/lessac-medium.ckpt"
CFG="/mnt/c/Users/user/Documents/VoiceModel/configs/train_ne.yaml"
CKPTDIR="$RUN/ckpts"
LAST="$CKPTDIR/last.ckpt"
mkdir -p "$RUN/cache" "$CKPTDIR"

COMMON=(
  fit
  --config "$CFG"
  --data.voice_name ne_stageA
  --data.csv_path "$DATA/metadata.csv"
  --data.audio_dir "$DATA/wavs"
  --data.espeak_voice ne
  --data.cache_dir "$RUN/cache"
  --data.config_path "$RUN/config.json"
  --data.batch_size 2
  --model.sample_rate 22050
  --model.num_speakers 20
  --trainer.accelerator gpu
  --trainer.devices 1
  --trainer.precision 32-true
  --trainer.max_epochs 2000
  --trainer.default_root_dir "$RUN"
  --trainer.num_sanity_val_steps 0
  --trainer.limit_val_batches 0
)

for attempt in $(seq 1 300); do
  echo "############ ATTEMPT $attempt ############"
  if [ -f "$LAST" ]; then
    echo ">>> resuming from $LAST"
    nice -n 10 python -m piper.train "${COMMON[@]}" --ckpt_path "$LAST"
  else
    echo ">>> fresh start (vocoder warm-start)"
    nice -n 10 python -m piper.train "${COMMON[@]}" --model.vocoder_warmstart_ckpt "$CKPT"
  fi
  code=$?
  if [ "$code" -eq 0 ]; then echo ">>> TRAINING COMPLETED NORMALLY"; break; fi
  echo ">>> attempt $attempt exited code $code (intermittent CUDA fault); restarting in 8s..."
  sleep 8
done
echo ">>> auto-resume loop ended"
