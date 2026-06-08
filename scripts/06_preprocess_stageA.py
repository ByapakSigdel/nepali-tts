#!/usr/bin/env python3
"""
Phase 2 — preprocess Stage A (SLR43 + SLR143) into a clean, uniform training set.

For every clip:  load -> mono -> resample 22.05kHz -> peak-normalize -> trim silence -> write.
Builds one master metadata file:  wavs/<name>.wav | <speaker> | <text>
Output: ~/voicemodel/data/processed/{wavs/, metadata.csv}
Re-runnable: skips clips already written.
"""
import os
import csv
import glob
import collections

import torch
import torchaudio
import soundfile as sf
import numpy as np

HOME = os.path.expanduser("~")
RAW = os.path.join(HOME, "voicemodel", "data", "raw")
OUT = os.path.join(HOME, "voicemodel", "data", "processed")
WAVOUT = os.path.join(OUT, "wavs")
os.makedirs(WAVOUT, exist_ok=True)
TARGET_SR = 22050

_resamplers = {}


def get_resampler(orig_sr):
    if orig_sr not in _resamplers:
        _resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, TARGET_SR)
    return _resamplers[orig_sr]


def trim_silence(y, sr, thresh=0.01, pad_ms=40):
    """Trim leading/trailing near-silence (y is peak-normalized, so thresh is relative)."""
    idx = np.where(np.abs(y) > thresh)[0]
    if len(idx) == 0:
        return y
    pad = int(sr * pad_ms / 1000)
    return y[max(0, idx[0] - pad): min(len(y), idx[-1] + pad)]


def process_clip(src_wav, dst_wav):
    y, sr = sf.read(src_wav, dtype="float32")     # soundfile load (no codec dep)
    if y.ndim > 1:                                 # to mono
        y = y.mean(axis=1)
    if sr != TARGET_SR:                            # resample via torchaudio math (no codec dep)
        t = torch.from_numpy(np.ascontiguousarray(y))
        y = torchaudio.functional.resample(t, sr, TARGET_SR).numpy()
    peak = float(np.max(np.abs(y))) or 1.0
    y = (y / peak) * 0.95                          # peak-normalize
    y = trim_silence(y, TARGET_SR)
    if len(y) < int(0.2 * TARGET_SR):              # skip <0.2s (junk)
        return None
    sf.write(dst_wav, y, TARGET_SR)
    return len(y) / TARGET_SR


def gather_records():
    """Yield (src_wav_path, speaker, text, out_name)."""
    recs = []
    # --- SLR43: line_index.tsv = <id>\t<text>; speaker = first two id tokens; wav in wavs/<id>.wav
    d43 = os.path.join(RAW, "slr43", "ne_np_female")
    for line in open(os.path.join(d43, "line_index.tsv"), encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        uid, text = parts[0].strip(), parts[-1].strip()
        spk = "_".join(uid.split("_")[:-1])       # e.g. nep_0546
        src = os.path.join(d43, "wavs", uid + ".wav")
        recs.append((src, spk, text, f"{spk}__{uid}.wav"))
    # --- SLR143: FemaleVoice.tsv / MaleVoice.tsv with header audio_id\tsentence; wav in <audio_id>.wav
    d143 = os.path.join(RAW, "slr143", "male-female-data")
    for tsv, spk in [("FemaleVoice.tsv", "slr143_female"), ("MaleVoice.tsv", "slr143_male")]:
        path = os.path.join(d143, tsv)
        if not os.path.exists(path):
            continue
        for i, row in enumerate(csv.reader(open(path, encoding="utf-8"), delimiter="\t")):
            if i == 0 and row and row[0].lower() == "audio_id":
                continue                          # skip header
            if len(row) < 2:
                continue
            aid, text = row[0].strip(), row[-1].strip()
            src = os.path.join(d143, aid + ".wav")
            recs.append((src, spk, text, f"{spk}__{aid}.wav"))
    return recs


def main():
    recs = gather_records()
    print(f">>> {len(recs)} candidate clips")
    meta_path = os.path.join(OUT, "metadata.csv")
    spk_counts = collections.Counter()
    spk_dur = collections.Counter()
    total = 0.0
    written = skipped = missing = 0
    with open(meta_path, "w", encoding="utf-8", newline="") as mf:
        for n, (src, spk, text, out_name) in enumerate(recs, 1):
            if not os.path.exists(src):
                missing += 1
                continue
            dst = os.path.join(WAVOUT, out_name)
            try:
                if os.path.exists(dst):
                    dur = sf.info(dst).frames / TARGET_SR
                else:
                    dur = process_clip(src, dst)
                if dur is None:
                    skipped += 1
                    continue
            except Exception as e:
                skipped += 1
                print(f"  ! {out_name}: {e}")
                continue
            mf.write(f"{out_name}|{spk}|{text}\n")
            spk_counts[spk] += 1
            spk_dur[spk] += dur
            total += dur
            written += 1
            if n % 400 == 0:
                print(f"  ...{n}/{len(recs)}  ({written} written)")

    print(f"\n>>> DONE: {written} written, {skipped} skipped, {missing} missing")
    print(f">>> total audio: {total/3600:.2f} h across {len(spk_counts)} speakers")
    print(f">>> metadata: {meta_path}")
    print(">>> per-speaker (clips, minutes):")
    for spk, c in spk_counts.most_common():
        print(f"     {spk:16s} {c:5d} clips  {spk_dur[spk]/60:6.1f} min")


if __name__ == "__main__":
    main()
