#!/usr/bin/env python3
"""Checkpoint guardian for the Stage-B run.

An unclean shutdown can truncate whatever the trainer is writing (we just lost
662 cache clips that way). If it truncates last.ckpt, the auto-resume loop would
crash-loop on it and DAYS of training would be gone. This guardian runs alongside
training and keeps a rotating set of *validated, disk-synced* backups of last.ckpt,
so there is always a recent known-good checkpoint to fall back to.

It also calls os.sync() every cycle, bounding how many dirty pages (cache writes
and checkpoint writes) an unclean shutdown can lose to roughly one INTERVAL.
"""
import glob
import os
import shutil
import sys
import time

import torch

CK = os.path.expanduser("~/voicemodel/models/ne_stageB/ckpts")
BK = os.path.join(CK, "backups")
LAST = os.path.join(CK, "last.ckpt")
KEEP = 4          # rotating validated backups to retain
INTERVAL = 300    # seconds between checks (~5 min worst-case loss window)

os.makedirs(BK, exist_ok=True)


def loadable(path):
    """True if the checkpoint can be opened without error (catches truncation)."""
    try:
        try:
            torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:  # older torch without mmap kwarg
            torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def log(msg):
    print(f"[guardian] {msg}", flush=True)


log(f"watching {LAST}; keeping {KEEP} backups in {BK}; every {INTERVAL}s")
last_saved_mtime = 0.0
while True:
    try:
        if os.path.exists(LAST):
            mtime = os.path.getmtime(LAST)
            if mtime != last_saved_mtime:
                tmp = os.path.join(BK, f".tmp_{int(mtime)}.ckpt")
                shutil.copyfile(LAST, tmp)          # copy first, validate the copy (no TOCTOU race)
                if loadable(tmp):
                    final = os.path.join(BK, f"ckpt_{int(mtime)}.ckpt")
                    os.replace(tmp, final)
                    os.sync()                        # flush backup to the vhdx
                    last_saved_mtime = mtime
                    backups = sorted(
                        glob.glob(os.path.join(BK, "ckpt_*.ckpt")),
                        key=os.path.getmtime, reverse=True,
                    )
                    for old in backups[KEEP:]:
                        os.remove(old)
                    log(f"backed up step-mtime {int(mtime)}; {len(backups[:KEEP])} kept")
                else:
                    os.remove(tmp)                   # caught a torn write; try again next cycle
                    log("last.ckpt not loadable yet (mid-write?), skipping")
        os.sync()                                    # bound dirty-page loss even before first ckpt
    except Exception as e:
        log(f"error: {e}")
    time.sleep(INTERVAL)
