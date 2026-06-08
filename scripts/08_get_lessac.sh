#!/usr/bin/env bash
# Download the Piper en_US-lessac-medium checkpoint (846MB) used to vocoder-warmstart our model.
set -u
mkdir -p ~/voicemodel/checkpoints
cd ~/voicemodel/checkpoints
URL="https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"
echo ">>> downloading lessac-medium.ckpt (846MB)..."
curl -L -C - --retry 8 --retry-delay 5 -sS -o lessac-medium.ckpt "$URL"
echo ">>> done:"
ls -lh lessac-medium.ckpt
echo ">>> LESSAC CHECKPOINT READY"
