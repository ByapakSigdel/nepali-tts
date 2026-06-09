#!/usr/bin/env python3
"""Extract the generator (model_g.*) weights from a Lightning VITS checkpoint into a small snapshot,
for later model-soup / EMA-style export. Generator-only keeps each snapshot ~150 MB (vs 882 MB full
checkpoints), so a dozen accumulate cheaply. Inference-only — never touches training.

  usage: python snapshot_generator.py [CKPT] [OUT_DIR]
"""
import os
import sys
import torch

ckpt = sys.argv[1] if len(sys.argv) > 1 else "/home/byapak/voicemodel/models/ne_stageA/ckpts/last.ckpt"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/byapak/voicemodel/models/ne_stageA/gen_snapshots"
os.makedirs(out_dir, exist_ok=True)

full = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
epoch = int(full.get("epoch", -1))
gen = {k: v.detach().clone() for k, v in full["state_dict"].items() if k.startswith("model_g.")}
assert gen, "no model_g.* tensors found — wrong checkpoint layout?"

out = os.path.join(out_dir, f"gen_ep{epoch:04d}.pt")
tmp = out + ".tmp"
torch.save({"epoch": epoch, "model_g": gen}, tmp)
os.replace(tmp, out)
print(f"snapshot saved: {out}  ({len(gen)} tensors, epoch {epoch}, {os.path.getsize(out) / 1e6:.0f} MB)")
