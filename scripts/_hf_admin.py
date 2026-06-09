#!/usr/bin/env python3
"""One-shot setup helper: identify the HF account and create the private relay repos.

Run inside the WSL venv. Reads the token from ~/voicemodel/.hf_token, creates a private
model repo (for the checkpoint baton) and a private dataset repo (for the training audio),
and records the checkpoint repo id in ~/voicemodel/.hf_repo so the relay scripts can find it.
"""
import os
from huggingface_hub import HfApi

VM = os.path.expanduser("~/voicemodel")
token = open(os.path.join(VM, ".hf_token")).read().strip()
api = HfApi(token=token)

who = api.whoami()
user = who["name"]
print("HF_USER:", user)

ckpt_repo = f"{user}/nepali-tts-ckpt"
data_repo = f"{user}/nepali-tts-data"

api.create_repo(ckpt_repo, repo_type="model", private=True, exist_ok=True)
api.create_repo(data_repo, repo_type="dataset", private=True, exist_ok=True)

with open(os.path.join(VM, ".hf_repo"), "w") as f:
    f.write(ckpt_repo)

print("CKPT_REPO:", ckpt_repo, "(private model repo — the baton)")
print("DATA_REPO:", data_repo, "(private dataset repo — training audio)")
print("wrote", os.path.join(VM, ".hf_repo"))
