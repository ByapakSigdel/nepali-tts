#!/usr/bin/env python3
"""
Generate paper figures from real data:
  pipeline.png       — system block diagram (text -> frontend -> VITS -> ONNX -> audio)
  mel_compare.png    — mel-spectrogram of a ground-truth clip vs the model synthesizing the SAME text
  duration_hist.png  — clip-duration distribution of the Stage-A corpus
Outputs into docs/.
"""
import os
import glob
import tempfile
import subprocess

import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
PROC = f"{HOME}/voicemodel/data/processed"
OUT = "/mnt/c/Users/user/Documents/VoiceModel/docs"
MODEL = f"{HOME}/voicemodel/frontend/model.onnx"
if not os.path.exists(MODEL):
    MODEL = f"{HOME}/voicemodel/_sample/ne.onnx"
os.makedirs(OUT, exist_ok=True)


def pipeline_fig():
    boxes = ["Nepali text\n(Devanagari or\nromanized)", "Frontend\ntranslit +\nespeak-ng G2P",
             "VITS\nmulti-speaker", "ONNX\nonnxruntime\n(CPU)", "Audio\n22.05 kHz"]
    centers = [0.1, 0.3, 0.5, 0.7, 0.9]
    bw, bh = 0.16, 0.5
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for c, b in zip(centers, boxes):
        ax.add_patch(plt.Rectangle((c - bw / 2, 0.5 - bh / 2), bw, bh,
                                   facecolor="#dce6f5", edgecolor="#33558c", lw=1.5))
        ax.text(c, 0.5, b, ha="center", va="center", fontsize=8.5)
    for i in range(len(centers) - 1):
        ax.annotate("", xy=(centers[i + 1] - bw / 2, 0.5), xytext=(centers[i] + bw / 2, 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333"))
    ax.set_title("System pipeline: Nepali text → fully-offline speech", fontsize=11)
    fig.savefig(f"{OUT}/pipeline.png", dpi=150, bbox_inches="tight")
    print("saved pipeline.png")


def _mel(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    S = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=sr, n_mels=80)
    return librosa.power_to_db(S, ref=np.max), sr


def mel_compare():
    rows = [l.rstrip("\n").split("|") for l in open(f"{PROC}/metadata.csv", encoding="utf-8") if l.strip()]
    pick = None
    for name, spk, text in rows:
        w = f"{PROC}/wavs/{name}"
        if os.path.exists(w):
            d = sf.info(w).frames / sf.info(w).samplerate
            if 2.5 < d < 4.5:
                pick = (w, text); break
    if not pick:
        print("no suitable clip for mel_compare"); return
    real_wav, text = pick
    gen_wav = tempfile.mktemp(suffix=".wav")
    subprocess.run(["python", "-m", "piper", "-m", MODEL, "-s", "0", "-f", gen_wav],
                   input=(text + "\n").encode("utf-8"), stderr=subprocess.DEVNULL)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2))
    for ax, path, title in [(axes[0], real_wav, "Ground truth (training clip)"),
                            (axes[1], gen_wav, "Synthesized (current model)")]:
        if not os.path.exists(path):
            continue
        Sdb, sr = _mel(path)
        librosa.display.specshow(Sdb, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_title(title, fontsize=9)
    fig.suptitle("Mel-spectrogram: same utterance — ground truth vs. synthesized", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mel_compare.png", dpi=140)
    print("saved mel_compare.png")


def duration_hist():
    durs = []
    for w in glob.glob(f"{PROC}/wavs/*.wav"):
        try:
            durs.append(sf.info(w).frames / sf.info(w).samplerate)
        except Exception:
            pass
    if not durs:
        print("no clips for duration_hist"); return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(durs, bins=40, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("clip duration (s)"); ax.set_ylabel("count")
    ax.set_title(f"Stage-A clip durations ({len(durs)} clips, total {sum(durs)/3600:.2f} h, "
                 f"median {np.median(durs):.1f} s)")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(f"{OUT}/duration_hist.png", dpi=140)
    print("saved duration_hist.png")


if __name__ == "__main__":
    pipeline_fig()
    duration_hist()
    mel_compare()
    print("DONE")
