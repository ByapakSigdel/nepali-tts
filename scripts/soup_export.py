#!/usr/bin/env python3
"""Model-soup / EMA-style export: average the generator weights across recent snapshots and export
ONNX. Inference-only — never touches the training run. Averaging late-plateau checkpoints from one
run (a.k.a. checkpoint averaging / SWA / EMA) is a well-established, low-risk way to get a smoother,
slightly higher-quality voice than any single checkpoint (Yazici et al. 2019; Izmailov et al. SWA).

  usage: python soup_export.py [SNAP_DIR] [BASE_CKPT] [OUT_ONNX] [LAST_K]
         LAST_K = average only the most recent K snapshots (0 = all). Use the latest few (tightest
         basin) if early snapshots are too far back.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import torch

snap_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/byapak/voicemodel/models/ne_stageA/gen_snapshots"
base_ckpt = sys.argv[2] if len(sys.argv) > 2 else "/home/byapak/voicemodel/models/ne_stageA/ckpts/last.ckpt"
out_onnx = sys.argv[3] if len(sys.argv) > 3 else "/home/byapak/voicemodel/_sample/ne_soup.onnx"
last_k = int(sys.argv[4]) if len(sys.argv) > 4 else 0
cfg = "/home/byapak/voicemodel/models/ne_stageA/config.json"

snaps = sorted(glob.glob(os.path.join(snap_dir, "gen_ep*.pt")))
if last_k > 0:
    snaps = snaps[-last_k:]
assert snaps, f"no snapshots in {snap_dir} — run snapshot_generator.py first"
print(f">>> souping {len(snaps)} generator snapshot(s):")
for s in snaps:
    print("   ", os.path.basename(s))

# Average model_g tensors across snapshots (accumulate in float64 for numerical safety).
acc, n = None, len(snaps)
for s in snaps:
    g = torch.load(s, map_location="cpu", weights_only=False)["model_g"]
    if acc is None:
        acc = {k: v.double().clone() for k, v in g.items()}
    else:
        for k in acc:
            acc[k] += g[k].double()
souped = {k: (v / n).to(torch.float32) for k, v in acc.items()}

# Write the souped generator into a copy of the base checkpoint, then export ONNX from it.
full = torch.load(base_ckpt, map_location="cpu", weights_only=False)
replaced = missing = 0
for k in list(full["state_dict"].keys()):
    if k.startswith("model_g."):
        if k in souped:
            full["state_dict"][k] = souped[k]
            replaced += 1
        else:
            missing += 1
print(f">>> replaced {replaced} generator tensors ({missing} not in snapshot — kept from base)")
tmp_ckpt = tempfile.mktemp(suffix=".ckpt")
torch.save(full, tmp_ckpt)
try:
    subprocess.run([sys.executable, "-m", "piper.train.export_onnx",
                    "--checkpoint", tmp_ckpt, "--output-file", out_onnx], check=True)
    shutil.copyfile(cfg, out_onnx + ".json")
    print(f">>> souped ONNX exported: {out_onnx}")
finally:
    os.remove(tmp_ckpt)
