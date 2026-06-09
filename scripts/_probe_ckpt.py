#!/usr/bin/env python3
"""Probe a Lightning VITS checkpoint: top-level keys, net_g/net_d param counts, speaker-embedding
key, epoch. Grounds the snapshot/soup scripts (and future DiLoCo) in the real state_dict layout."""
import sys
import torch

p = sys.argv[1] if len(sys.argv) > 1 else "/home/byapak/voicemodel/models/ne_stageA/ckpts/last.ckpt"
sd = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
print("top-level keys:", list(sd.keys()))
ks = list(sd["state_dict"].keys())
print("total state_dict tensors:", len(ks))
print("net_g. tensors:", sum(k.startswith("net_g.") for k in ks))
print("net_d. tensors:", sum(k.startswith("net_d.") for k in ks))
print("emb_g keys:", [k for k in ks if "emb_g" in k][:5])
print("sample net_g keys:", [k for k in ks if k.startswith("net_g.")][:5])
print("epoch:", sd.get("epoch"), "global_step:", sd.get("global_step"))
print("optimizer_states:", len(sd.get("optimizer_states", [])))
