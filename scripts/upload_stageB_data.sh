#!/usr/bin/env bash
# Package Stage-B processed data (wavs + metadata.csv) and upload to the HF dataset
# repo as processed_b.tar.gz. Long-running; safe to re-run (skips tar if present,
# upload retries with backoff).
# Run via:  wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/Documents/VoiceModel/scripts/upload_stageB_data.sh
set -euo pipefail
source ~/voicemodel/.venv/bin/activate

export HF_TOKEN=$(cat ~/voicemodel/.hf_token)
DATA_REPO="byapaksigdel/nepali-tts-data"
SRC=~/voicemodel/data/processed_b
TAR=~/voicemodel/data/processed_b.tar.gz

if [ ! -f "$TAR" ]; then
  echo "== creating $TAR (gzip -1; wavs barely compress, speed matters) =="
  tar -C "$(dirname "$SRC")" -I 'gzip -1' -cf "$TAR" "$(basename "$SRC")"
else
  echo "== $TAR already exists, skipping tar =="
fi
ls -lh "$TAR"

echo "== uploading to $DATA_REPO (retries with backoff) =="
python - "$TAR" "$DATA_REPO" <<'PY'
import sys, time
from huggingface_hub import HfApi
import os

tar, repo = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for attempt in range(1, 9):
    try:
        api.upload_file(
            path_or_fileobj=tar,
            path_in_repo="processed_b.tar.gz",
            repo_id=repo,
            repo_type="dataset",
            commit_message="Stage B data: SLR54 processed (22.05kHz wavs + metadata)",
        )
        print("UPLOAD OK")
        break
    except Exception as e:
        print(f"attempt {attempt} failed: {e}; retrying in {30*attempt}s", flush=True)
        time.sleep(30 * attempt)
else:
    sys.exit("UPLOAD FAILED after 8 attempts")
PY
echo "== data upload complete =="
