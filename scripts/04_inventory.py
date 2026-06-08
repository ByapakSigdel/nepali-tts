#!/usr/bin/env python3
"""Phase 1 — inventory the downloaded Nepali datasets: clips, speakers, hours, sample rates."""
import os
import glob
import collections
import soundfile as sf

HOME = os.path.expanduser("~")
RAW = os.path.join(HOME, "voicemodel", "data", "raw")


def scan_wavs(paths):
    """Return (total_seconds, sample_rate_counts, channel_counts, n_ok, n_bad)."""
    total = 0.0
    srs = collections.Counter()
    chs = collections.Counter()
    ok = bad = 0
    for p in paths:
        try:
            info = sf.info(p)
            total += info.frames / info.samplerate
            srs[info.samplerate] += 1
            chs[info.channels] += 1
            ok += 1
        except Exception:
            bad += 1
    return total, srs, chs, ok, bad


def fmt_hours(sec):
    return f"{sec/3600:.2f} h ({sec/60:.0f} min)"


# ---------------- SLR43 ----------------
print("=" * 64)
print("SLR43 — ne_np_female (high-quality multi-speaker female TTS)")
print("=" * 64)
d43 = os.path.join(RAW, "slr43", "ne_np_female")
tsv43 = os.path.join(d43, "line_index.tsv")
rows = [l.rstrip("\n").split("\t") for l in open(tsv43, encoding="utf-8") if l.strip()]
print(f"transcript rows : {len(rows)}")
print("sample rows     :")
for r in rows[:4]:
    print("   ", r)
# utterance id is column 0 (Google sets sometimes have a leading blank col); detect
idcol = 0 if rows and rows[0][0] else 1
ids = [r[idcol] for r in rows]
# speaker = id minus last underscore token (Google convention: <lang><gender>_<speaker>_<utt>)
speakers = collections.Counter("_".join(i.split("_")[:-1]) for i in ids)
print(f"unique speakers : {len(speakers)}")
spc = sorted(speakers.values())
if spc:
    print(f"utts/speaker    : min {spc[0]}, median {spc[len(spc)//2]}, max {spc[-1]}")
print("sample speakers :", list(speakers.items())[:5])
wavs43 = glob.glob(os.path.join(d43, "wavs", "*.wav"))
print(f"wav files       : {len(wavs43)}")
tot, srs, chs, ok, bad = scan_wavs(wavs43)
print(f"total audio     : {fmt_hours(tot)}  (ok {ok}, unreadable {bad})")
print(f"sample rates    : {dict(srs)}")
print(f"channels        : {dict(chs)}")

# ---------------- SLR143 ----------------
print()
print("=" * 64)
print("SLR143 — male-female-data (Nepali TTS, male + female)")
print("=" * 64)
d143 = os.path.join(RAW, "slr143", "male-female-data")
tsvs = glob.glob(os.path.join(d143, "*.tsv"))
print(f"tsv files       : {[os.path.basename(t) for t in tsvs]}")
for t in tsvs:
    tr = [l.rstrip("\n").split("\t") for l in open(t, encoding="utf-8") if l.strip()]
    print(f"  {os.path.basename(t)}: {len(tr)} rows; sample: {tr[:2]}")
wavs143 = glob.glob(os.path.join(d143, "*.wav"))
print(f"wav files       : {len(wavs143)}")
tot2, srs2, chs2, ok2, bad2 = scan_wavs(wavs143)
print(f"total audio     : {fmt_hours(tot2)}  (ok {ok2}, unreadable {bad2})")
print(f"sample rates    : {dict(srs2)}")
print(f"channels        : {dict(chs2)}")

# ---------------- Summary ----------------
print()
print("=" * 64)
print("COMBINED (Stage A)")
print("=" * 64)
print(f"total clips     : {len(wavs43) + len(wavs143)}")
print(f"total audio     : {fmt_hours(tot + tot2)}")
print(f"SLR43 speakers  : {len(speakers)} (female)  + SLR143 male/female")
