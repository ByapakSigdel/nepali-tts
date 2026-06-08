#!/usr/bin/env bash
# Diagnose GPU usage: (A) can the GPU hit ~100%? (B) what util does training actually get?
set -uo pipefail
# shellcheck disable=SC1091
source ~/voicemodel/.venv/bin/activate
export PYTORCH_JIT=0

echo "================ PART A: pure GPU saturation test ================"
python - <<'PY' &
import torch, time
a = torch.randn(4096, 4096, device="cuda")
t0 = time.time()
while time.time() - t0 < 14:
    a = (a @ a)
    a = a / (a.abs().max() + 1e-6)
torch.cuda.synchronize()
PY
LPID=$!
echo "(sampling GPU sm%/mem% for ~12s while a matmul loop runs)"
nvidia-smi dmon -s u -c 12 2>/dev/null
wait $LPID 2>/dev/null
echo ""

echo "================ PART B: actual TRAINING GPU utilization ================"
cd ~/voicemodel/piper1-gpl
VM="$HOME/voicemodel"; DATA="$VM/data/processed"; RUN="$VM/models/ne_stageA"
( python -m piper.train fit \
    --config /mnt/c/Users/user/Documents/VoiceModel/configs/train_ne.yaml \
    --data.csv_path "$DATA/metadata.csv" --data.audio_dir "$DATA/wavs" --data.espeak_voice ne \
    --data.cache_dir "$RUN/cache" --data.config_path "$RUN/config.json" --data.batch_size 4 \
    --model.sample_rate 22050 --model.num_speakers 20 --trainer.accelerator gpu --trainer.devices 1 \
    --trainer.precision 32-true --trainer.num_sanity_val_steps 0 --trainer.limit_val_batches 0 \
    --trainer.default_root_dir "$RUN" --ckpt_path "$RUN/ckpts/last.ckpt" > /tmp/diag_train.log 2>&1 ) &
TPID=$!
echo "(loading model ~30s, then sampling GPU sm%/mem% during real training steps)"
sleep 32
nvidia-smi dmon -s u -c 18 2>/dev/null
kill "$TPID" 2>/dev/null
pkill -9 -f "[p]iper.train" 2>/dev/null
sleep 1
echo ""
echo "--- dataloader workers in use? (low workers = GPU starved) ---"
grep -iE "does not have many workers|num_workers" /tmp/diag_train.log | head -2 || echo "(no worker warning found)"
echo ">>> DIAG DONE"
