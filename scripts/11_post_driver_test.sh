#!/usr/bin/env bash
# Run AFTER updating the NVIDIA driver (+ reboot or `wsl --shutdown`).
# Verifies the GPU, then runs a SHORT training smoke test (to global step 400) to check whether
# the intermittent backward-pass CUDA fault is gone. Resumes from last.ckpt (no progress lost).
set -uo pipefail
# shellcheck disable=SC1091
source ~/voicemodel/.venv/bin/activate
export PYTORCH_JIT=0

echo "================ DRIVER + GPU ================"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader
python - <<'PY'
import torch
cap = torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0), "| sm_%d%d" % cap)
PY

echo "================ TRAINING SMOKE TEST (to step 400) ================"
cd ~/voicemodel/piper1-gpl
VM="$HOME/voicemodel"; DATA="$VM/data/processed"; RUN="$VM/models/ne_stageA"
python -m piper.train fit \
  --config /mnt/c/Users/user/Documents/VoiceModel/configs/train_ne.yaml \
  --data.voice_name ne_stageA --data.csv_path "$DATA/metadata.csv" --data.audio_dir "$DATA/wavs" \
  --data.espeak_voice ne --data.cache_dir "$RUN/cache" --data.config_path "$RUN/config.json" \
  --data.batch_size 4 --model.sample_rate 22050 --model.num_speakers 20 \
  --trainer.accelerator gpu --trainer.devices 1 --trainer.precision 32-true \
  --trainer.max_steps 400 --trainer.num_sanity_val_steps 0 --trainer.limit_val_batches 0 \
  --trainer.default_root_dir "$RUN" --ckpt_path "$RUN/ckpts/last.ckpt"

echo ">>> If this reached step 400 with NO 'CUDA error', the driver update FIXED the fault."
echo ">>> Then run the full grind: bash scripts/10_train_autoresume.sh"
