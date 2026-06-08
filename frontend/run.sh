#!/usr/bin/env bash
# Launch the offline Nepali TTS web app.
#   wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/Documents/VoiceModel/frontend/run.sh
# Then open http://localhost:7860 in your Windows browser.
# Set REFRESH=0 to skip re-exporting the latest checkpoint (faster startup).
set -uo pipefail
# shellcheck disable=SC1091
source ~/voicemodel/.venv/bin/activate
export PYTHONUNBUFFERED=1

# Ensure frontend deps (idempotent; both are torch-free so they won't disturb torch 2.11).
python -c "import gradio, indic_transliteration" 2>/dev/null || pip install -q gradio "indic-transliteration==2.3.82"

FE="$HOME/voicemodel/frontend"; mkdir -p "$FE"
RUN="$HOME/voicemodel/models/ne_stageA"

if [ "${REFRESH:-1}" = "1" ] && [ -f "$RUN/ckpts/last.ckpt" ]; then
  echo ">>> exporting latest checkpoint to a fresh frontend model (best-effort)..."
  cp "$RUN/ckpts/last.ckpt" "$FE/cur.ckpt"
  if python -m piper.train.export_onnx --checkpoint "$FE/cur.ckpt" --output-file "$FE/model.onnx" >/dev/null 2>&1; then
    cp "$RUN/config.json" "$FE/model.onnx.json"
    echo ">>> fresh model ready: $FE/model.onnx"
  else
    echo ">>> export skipped/failed; the app will use the existing model."
  fi
fi

export NE_TTS_MODEL="$FE/model.onnx"
echo ">>> Launching Nepali TTS web app  ->  open http://localhost:7860"
python /mnt/c/Users/user/Documents/VoiceModel/frontend/app.py
