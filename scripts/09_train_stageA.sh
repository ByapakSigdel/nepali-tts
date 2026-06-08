#!/usr/bin/env bash
# Phase 4 — train the multi-speaker Nepali VITS model on Stage A data (piper1-gpl).
# Tunable via env vars: BATCH, MAXEPOCHS, WARM (vocoder|ckpt|none), PRECISION.
set -euo pipefail
# shellcheck disable=SC1091
source ~/voicemodel/.venv/bin/activate
cd ~/voicemodel/piper1-gpl

# Blackwell (sm_120) + torch 2.11: VITS's @torch.jit.script fused op fails in the
# TorchScript interpreter (misreported as a 20MiB OOM). Run those ops eager instead.
export PYTORCH_JIT=0
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here — it uses CUDA
# virtual-memory APIs that are broken under WSL2 (causes bogus "16 billion GB in use" OOM).

VM="$HOME/voicemodel"
DATA="$VM/data/processed"
RUN="$VM/models/ne_stageA"
CKPT="$VM/checkpoints/lessac-medium.ckpt"
mkdir -p "$RUN/cache"

BATCH="${BATCH:-4}"            # small for 8GB VRAM + WSL stability
MAXEPOCHS="${MAXEPOCHS:-2000}"
WARM="${WARM:-vocoder}"        # vocoder | ckpt | none
# Precision journey: bf16 -> cuFFT can't do bf16 (STFT crash); fp16 -> experimental ComplexHalf
# FFT was unstable on Blackwell (random CUDA fault). fp32 = fully supported, most stable.
PRECISION="${PRECISION:-32-true}"

ARGS=(
  fit
  --data.voice_name "ne_stageA"
  --data.csv_path "$DATA/metadata.csv"
  --data.audio_dir "$DATA/wavs"
  --data.espeak_voice "ne"
  --data.cache_dir "$RUN/cache"
  --data.config_path "$RUN/config.json"
  --data.batch_size "$BATCH"
  --model.sample_rate 22050
  --model.num_speakers 20
  --trainer.accelerator gpu
  --trainer.devices 1
  --trainer.precision "$PRECISION"
  --trainer.max_epochs "$MAXEPOCHS"
  --trainer.default_root_dir "$RUN"
)

case "$WARM" in
  vocoder) ARGS+=(--model.vocoder_warmstart_ckpt "$CKPT") ;;
  ckpt)    ARGS+=(--ckpt_path "$CKPT") ;;
  none)    ;;
esac

echo ">>> WARM=$WARM BATCH=$BATCH PRECISION=$PRECISION"
echo ">>> python -m piper.train ${ARGS[*]}"
python -m piper.train "${ARGS[@]}"
