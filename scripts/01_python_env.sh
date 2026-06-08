#!/usr/bin/env bash
# Phase 0 — Python environment + PyTorch (run inside WSL Ubuntu-24.04). No sudo needed.
# Recreates a clean venv (now that python3.12-venv is installed) and installs the
# Blackwell-compatible PyTorch build (CUDA 12.8 / sm_120).
set -euo pipefail

VM="$HOME/voicemodel"
echo ">>> (1/4) Recreating clean venv at $VM/.venv"
rm -rf "$VM/.venv"
python3 -m venv "$VM/.venv"
# shellcheck disable=SC1091
source "$VM/.venv/bin/activate"

echo ">>> (2/4) Upgrading pip/wheel/setuptools"
python -m pip install --upgrade pip wheel setuptools

echo ">>> (3/4) Installing PyTorch + torchaudio (cu128 for Blackwell sm_120)"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

echo ">>> (4/4) Verifying torch + GPU"
python - <<'PY'
import torch
print("torch version :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name      :", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("Compute cap   :", f"sm_{cap[0]}{cap[1]}  (Blackwell = sm_120)")
    x = torch.randn(2048, 2048, device="cuda")
    y = (x @ x).sum().item()  # forces a real GPU compute
    print("GPU matmul OK :", abs(y) > 0)
PY
echo ">>> ENV SETUP COMPLETE"
