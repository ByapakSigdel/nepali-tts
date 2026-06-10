#!/usr/bin/env bash
# Stage B trainer: SLR54 (~112h / 527 speakers), warm-started from the Stage-A epoch-274 model via the
# SMART warm-start (full generator + discriminator transfer; only the speaker table is fresh). Same
# auto-resume loop as Stage A: checkpoints every 100 steps, restarts from last.ckpt after any GPU fault.
set -uo pipefail
# shellcheck disable=SC1091
source ~/voicemodel/.venv/bin/activate
cd ~/voicemodel/piper1-gpl

export PYTORCH_JIT=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="/mnt/c/Users/user/Documents/VoiceModel/scripts:${PYTHONPATH:-}"

VM="$HOME/voicemodel"
DATA="$VM/data/processed_b"
RUN="$VM/models/ne_stageB"
WARM="$VM/checkpoints/stageB_warmstart.ckpt"   # smart warm-start source (epoch-274 snapshot)
CFG="/mnt/c/Users/user/Documents/VoiceModel/configs/train_ne_stageB.yaml"
CKPTDIR="$RUN/ckpts"
LAST="$CKPTDIR/last.ckpt"
mkdir -p "$RUN/cache" "$CKPTDIR"

COMMON=(
  fit
  --config "$CFG"
  --data.voice_name ne_stageB
  --data.csv_path "$DATA/metadata.csv"
  --data.audio_dir "$DATA/wavs"
  --data.espeak_voice ne
  --data.cache_dir "$RUN/cache"
  --data.config_path "$RUN/config.json"
  --data.batch_size ${BATCH:-4}
  --data.num_workers ${WORKERS:-8}
  --model.sample_rate 22050
  --model.num_speakers 527
  --trainer.accelerator gpu
  --trainer.devices 1
  --trainer.precision 32-true
  --trainer.max_epochs 100
  --trainer.default_root_dir "$RUN"
  --trainer.num_sanity_val_steps 0
  --trainer.limit_val_batches 0
  --trainer.log_every_n_steps 25
)

PICK="/mnt/c/Users/user/Documents/VoiceModel/scripts/pick_ckpt.py"
for attempt in $(seq 1 500); do
  echo "############ STAGE-B ATTEMPT $attempt ############"
  # pick_ckpt validates last.ckpt; if a shutdown truncated it, it falls back to the
  # newest valid guardian backup instead of crash-looping on a corrupt file.
  CK="$(python "$PICK")"
  if [ -n "$CK" ]; then
    echo ">>> resuming from $CK"
    python -m piper.train "${COMMON[@]}" --ckpt_path "$CK"
  else
    echo ">>> fresh start — SMART warm-start from $WARM (full generator + discriminator)"
    python -m piper.train "${COMMON[@]}" --model.vocoder_warmstart_ckpt "$WARM"
  fi
  code=$?
  if [ "$code" -eq 0 ]; then echo ">>> STAGE B COMPLETED NORMALLY"; break; fi
  echo ">>> attempt $attempt exited code $code (intermittent fault); restarting in 8s..."
  sleep 8
done
echo ">>> stage-B auto-resume loop ended"
