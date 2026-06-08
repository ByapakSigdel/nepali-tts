#!/usr/bin/env python3
"""
Plot training curves from all Lightning TensorBoard logs (merged across every restart/version).
Outputs PNGs into docs/ for the writeup.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tbparse import SummaryReader

LOGDIR = "/home/byapak/voicemodel/models/ne_stageA/lightning_logs"
OUTDIR = "/mnt/c/Users/user/Documents/VoiceModel/docs"
os.makedirs(OUTDIR, exist_ok=True)

print(">>> reading all tfevents (this merges every restart)...")
df = SummaryReader(LOGDIR, extra_columns={"dir_name"}).scalars
print(f">>> {len(df)} scalar points")
tags = sorted(df["tag"].unique())
print(">>> available metrics:")
for t in tags:
    print("     ", t)


def series(tag, smooth=1):
    d = df[df["tag"] == tag][["step", "value"]].dropna()
    d = d.sort_values("step").drop_duplicates("step", keep="last")
    s, v = d["step"].values, d["value"].values
    if smooth > 1 and len(v) >= smooth:
        import numpy as np
        v = np.convolve(v, np.ones(smooth) / smooth, mode="valid")
        s = s[smooth - 1:]
    return s, v


# ---- Figure 1: all loss components ----
loss_tags = [t for t in tags if "loss" in t.lower()]
if loss_tags:
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in loss_tags:
        s, v = series(t, smooth=5)
        if len(s):
            ax.plot(s, v, linewidth=1.1, label=t)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Nepali multi-speaker VITS — training losses (all runs merged)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUTDIR, "training_losses.png")
    fig.savefig(p, dpi=140)
    print(">>> saved", p)

# ---- Figure 2: headline generator/total loss ----
head = next((t for t in ("loss_g", "loss_gen_all", "loss_gen_total", "g_total", "loss", "train_loss") if t in tags), None)
if head is None and loss_tags:
    head = loss_tags[0]
if head:
    s, v = series(head, smooth=9)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(s, v, color="#1f77b4", linewidth=1.6)
    ax.set_xlabel("Training step")
    ax.set_ylabel(head)
    ax.set_title(f"Headline training loss ({head}) — lower = better")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUTDIR, "headline_loss.png")
    fig.savefig(p, dpi=140)
    print(">>> saved", p, f"| latest {head} = {v[-1]:.3f} at step {int(s[-1])}" if len(v) else "")

# ---- Figure 3: dataset speaker distribution ----
import collections
META = "/home/byapak/voicemodel/data/processed/metadata.csv"
if os.path.exists(META):
    cnt = collections.Counter()
    for line in open(META, encoding="utf-8"):
        p = line.rstrip("\n").split("|")
        if len(p) >= 2:
            cnt[p[1]] += 1
    if cnt:
        items = cnt.most_common()
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(range(len(names)), vals, color="#4c72b0")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("clips")
        ax.set_title(f"Stage-A dataset: clips per speaker ({len(names)} speakers, {sum(vals)} clips)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        pth = os.path.join(OUTDIR, "dataset_speakers.png")
        fig.savefig(pth, dpi=140)
        print(">>> saved", pth)

print(">>> PLOTTING DONE")
