#!/usr/bin/env bash
# Phase 4 — install piper1-gpl trainer into our venv WITHOUT disturbing torch 2.11+cu128.
# torch>=2,<3 is satisfied by 2.11, so pip should not touch it. We verify before/after.
set -euo pipefail
# shellcheck disable=SC1091
source ~/voicemodel/.venv/bin/activate
cd ~/voicemodel/piper1-gpl

echo ">>> ninja build tool (no sudo)"
pip install -q ninja

echo ">>> torch BEFORE:"
python -c "import torch; print(' ', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

echo ">>> installing piper1-gpl [train] ..."
pip install -e '.[train]'

echo ">>> torch AFTER (must be unchanged 2.11+cu128, cuda avail True):"
python -c "import torch; print(' ', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

echo ">>> building monotonic_align + C extensions"
bash ./build_monotonic_align.sh
python setup.py build_ext --inplace

echo ">>> import check"
python -c "import piper.train; print(' piper.train import OK')"
echo ">>> PIPER INSTALL COMPLETE"
