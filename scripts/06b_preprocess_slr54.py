#!/usr/bin/env python3
"""Stage B preprocessing: OpenSLR-54 (Large Nepali ASR, ~151h / 527 speakers) -> piper training format.
Reads FLACs straight from the zip shards (no 7GB extraction). Mirrors 06_preprocess_stageA.py:
  load -> mono -> resample to 22.05kHz -> peak-normalize -> trim silence -> write wav.
Output: ~/voicemodel/data/processed_b/{wavs/, metadata.csv}  (name|speaker|text, speaker = slr54_<id>).
Resumable: existing wavs are reused, metadata.csv is rebuilt each run so it always stays consistent.
"""
import collections
import glob
import io
import os
import zipfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

HOME = os.path.expanduser("~")
SLR54 = os.path.join(HOME, "voicemodel", "data", "raw", "slr54")
OUT = os.path.join(HOME, "voicemodel", "data", "processed_b")
WAVOUT = os.path.join(OUT, "wavs")
os.makedirs(WAVOUT, exist_ok=True)
TARGET_SR = 22050
MIN_DUR, MAX_DUR = 1.0, 12.0

_resampler = torchaudio.transforms.Resample(16000, TARGET_SR)  # SLR54 is 16kHz; cache the kernel


def resample(y, sr):
    if sr == TARGET_SR:
        return y
    t = torch.from_numpy(np.ascontiguousarray(y))
    if sr == 16000:
        return _resampler(t).numpy()
    return torchaudio.functional.resample(t, sr, TARGET_SR).numpy()


def trim_silence(y, sr, thresh=0.01, pad_ms=40):
    idx = np.where(np.abs(y) > thresh)[0]
    if len(idx) == 0:
        return y
    pad = int(sr * pad_ms / 1000)
    return y[max(0, idx[0] - pad): min(len(y), idx[-1] + pad)]


def process(data_bytes):
    y, sr = sf.read(io.BytesIO(data_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = resample(y, sr)
    peak = float(np.max(np.abs(y))) or 1.0
    y = (y / peak) * 0.95
    return trim_silence(y, TARGET_SR)


def load_tsv():
    z = zipfile.ZipFile(os.path.join(SLR54, "asr_nepali_0.zip"))
    tsv = z.read("asr_nepali/utt_spk_text.tsv").decode("utf-8")
    d = {}
    for line in tsv.splitlines():
        p = line.split("\t")
        if len(p) >= 3 and p[2].strip():
            d[p[0]] = (p[1], p[2].strip())
    return d


def main():
    meta = load_tsv()
    print(f">>> {len(meta)} transcripts loaded", flush=True)
    zips = sorted(glob.glob(os.path.join(SLR54, "asr_nepali_*.zip")))
    mf = open(os.path.join(OUT, "metadata.csv"), "w", encoding="utf-8")
    written = reused = skipped = 0
    spk_dur = collections.Counter()
    for zp in zips:
        try:
            z = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            print(f"  !! SKIPPING corrupt/truncated zip: {os.path.basename(zp)}", flush=True)
            continue
        for name in z.namelist():
            if not name.endswith(".flac"):
                continue
            uid = os.path.basename(name)[:-5]
            if uid not in meta:
                skipped += 1
                continue
            spk, text = meta[uid]
            spk_label = f"slr54_{spk}"
            out_name = f"{spk_label}__{uid}.wav"
            dst = os.path.join(WAVOUT, out_name)
            try:
                if os.path.exists(dst):
                    dur = sf.info(dst).frames / TARGET_SR
                    reused += 1
                else:
                    y = process(z.read(name))
                    dur = len(y) / TARGET_SR
                    if dur < MIN_DUR or dur > MAX_DUR:
                        skipped += 1
                        continue
                    sf.write(dst, y, TARGET_SR)
                    written += 1
            except Exception:
                skipped += 1
                continue
            mf.write(f"{out_name}|{spk_label}|{text}\n")
            spk_dur[spk_label] += dur
            done = written + reused
            if done % 2000 == 0:
                print(f"  ...{done} done ({written} new, {skipped} skipped), "
                      f"{sum(spk_dur.values())/3600:.1f}h so far", flush=True)
    mf.close()
    print(f"\n>>> DONE: {written} new, {reused} reused, {skipped} skipped", flush=True)
    print(f">>> total audio: {sum(spk_dur.values())/3600:.2f}h across {len(spk_dur)} speakers", flush=True)
    print(f">>> metadata: {os.path.join(OUT, 'metadata.csv')}", flush=True)


if __name__ == "__main__":
    main()
