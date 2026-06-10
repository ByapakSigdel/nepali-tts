#!/usr/bin/env python3
"""Print a *loadable* checkpoint path for the Stage-B auto-resume loop.

Preference order: last.ckpt, then the newest validated backup. If last.ckpt was
truncated by an unclean shutdown, we skip it instead of crash-looping on it.
Prints nothing (empty) if no valid checkpoint exists -> caller does a fresh start.
"""
import glob
import os
import sys

import torch

CK = os.path.expanduser("~/voicemodel/models/ne_stageB/ckpts")
candidates = [os.path.join(CK, "last.ckpt")] + sorted(
    glob.glob(os.path.join(CK, "backups", "ckpt_*.ckpt")),
    key=lambda p: os.path.getmtime(p),
    reverse=True,
)

for c in candidates:
    if not os.path.exists(c):
        continue
    try:
        try:
            torch.load(c, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            torch.load(c, map_location="cpu", weights_only=False)
        print(c)
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(f"[pick_ckpt] skipping unloadable {os.path.basename(c)}: {e}\n")
        continue

sys.exit(0)  # nothing valid -> empty output -> fresh start
