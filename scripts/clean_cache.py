#!/usr/bin/env python3
"""Scan the Stage-B cache for truncated/corrupt .pt files (left by an interrupted
write, e.g. an abrupt shutdown) and remove the whole clip so it gets re-cached
cleanly on the next run. The trainer's prepare_data regenerates any missing file."""
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

cache = os.path.expanduser("~/voicemodel/models/ne_stageB/cache")
files = glob.glob(os.path.join(cache, "**", "*.pt"), recursive=True)
print(f"scanning {len(files)} .pt files in {cache} ...", flush=True)


def check(p):
    try:
        torch.load(p, map_location="cpu", weights_only=False)
        return None
    except Exception as e:  # EOFError / UnpicklingError / RuntimeError on truncation
        return (p, type(e).__name__)


corrupt = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(check, p) for p in files]
    done = 0
    for fut in as_completed(futs):
        r = fut.result()
        done += 1
        if r:
            corrupt.append(r)
        if done % 20000 == 0:
            print(f"  {done}/{len(files)} checked, {len(corrupt)} corrupt", flush=True)

print(f"\ncorrupt files: {len(corrupt)}")
ids = set()
for p, err in corrupt:
    print(f"  {os.path.basename(p)}: {err}")
    ids.add(os.path.basename(p).split(".")[0])

removed = 0
for i in sorted(ids):
    for f in glob.glob(os.path.join(cache, i + ".*")):
        os.remove(f)
        removed += 1
print(f"removed {removed} files for {len(ids)} clip id(s); they will re-cache on next run")
