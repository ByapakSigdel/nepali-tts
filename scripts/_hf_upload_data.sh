#!/usr/bin/env bash
# One-off: pack the processed Stage-A dataset and upload it to the private HF dataset repo,
# so a Colab session can pull the exact same training audio.
set -uo pipefail
VM="$HOME/voicemodel"
source "$VM/.venv/bin/activate"
export HF_TOKEN="$(cat "$VM/.hf_token")"
export DATA_REPO="$(sed 's/-ckpt$/-data/' "$VM/.hf_repo")"
echo ">>> dataset repo: $DATA_REPO"
echo ">>> packing processed dataset (wavs + metadata)..."
tar czf /tmp/processed.tar.gz -C "$VM/data" processed
ls -lh /tmp/processed.tar.gz
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["DATA_REPO"]
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
print(">>> uploading processed.tar.gz ...", flush=True)
api.upload_file(
    path_or_fileobj="/tmp/processed.tar.gz",
    path_in_repo="processed.tar.gz",
    repo_id=repo, repo_type="dataset",
    commit_message="dataset: processed Stage-A audio (2739 clips, 22.05kHz)",
)
print(">>> DATASET UPLOADED to", repo, flush=True)
PY
rm -f /tmp/processed.tar.gz
echo ">>> data upload complete"
