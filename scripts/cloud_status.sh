#!/usr/bin/env bash
# Report the Stage-B relay repo state: remote epoch, lock, latest commits, PROGRESS.json.
set -uo pipefail
# shellcheck disable=SC1091
source ~/voicemodel/.venv/bin/activate
export HF_TOKEN="$(cat ~/voicemodel/.hf_token)"
export HF_REPO="byapaksigdel/nepali-tts-ckpt-b"
SYNC=/mnt/c/Users/user/Documents/VoiceModel/scripts/hf_sync.py

echo -n ">>> remote epoch: "; python "$SYNC" remote-epoch
python "$SYNC" status
python - <<'PY'
import os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ["HF_TOKEN"])
print(">>> latest commits:")
for c in api.list_repo_commits(os.environ["HF_REPO"])[:5]:
    print(f"     {c.created_at:%m-%d %H:%M}  {c.title}")
p = hf_hub_download(os.environ["HF_REPO"], "PROGRESS.json",
                    token=os.environ["HF_TOKEN"], force_download=True)
print(">>> PROGRESS.json:", open(p).read().strip())
PY
