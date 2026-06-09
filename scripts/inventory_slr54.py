#!/usr/bin/env python3
"""Inventory OpenSLR-54 (Large Nepali ASR set) for TTS curation planning, WITHOUT extracting 7GB.
Reads the master utt_spk_text.tsv (in shard 0) for the speaker/utterance distribution, and probes a
few FLACs for sample rate / duration. Prints how many speakers clear various clip-count thresholds."""
import collections
import io
import statistics
import zipfile

import soundfile as sf

SLR54 = "/home/byapak/voicemodel/data/raw/slr54"
Z0 = f"{SLR54}/asr_nepali_0.zip"

z = zipfile.ZipFile(Z0)
tsv = z.read("asr_nepali/utt_spk_text.tsv").decode("utf-8")
lines = [l for l in tsv.splitlines() if l.strip()]
print(f"utt_spk_text.tsv lines: {len(lines)}")
print("sample lines:")
for l in lines[:3]:
    print("   ", repr(l[:110]))

spk_counts = collections.Counter()
txt_lens = []
for l in lines:
    parts = l.split("\t")
    if len(parts) >= 3:
        spk_counts[parts[1]] += 1
        txt_lens.append(len(parts[2]))

counts = sorted(spk_counts.values(), reverse=True)
print(f"\ntotal utterances: {sum(counts)}")
print(f"unique speakers:  {len(counts)}")
print(f"clips/speaker:    max {counts[0]}, median {int(statistics.median(counts))}, min {counts[-1]}")
print(f"transcript chars: median {int(statistics.median(txt_lens))}")
print("\nspeakers clearing each clip-count threshold (candidate TTS voices):")
for thr in [50, 100, 200, 300, 500, 1000]:
    sel = [c for c in counts if c >= thr]
    print(f"  >= {thr:>4} clips: {len(sel):>4} speakers, covering {sum(sel):>7} utterances")

# probe a few FLACs for sample rate / duration (TTS fidelity depends on this)
srs, durs = collections.Counter(), []
seen = 0
for n in z.namelist():
    if n.endswith(".flac"):
        info = sf.info(io.BytesIO(z.read(n)))
        srs[info.samplerate] += 1
        durs.append(info.frames / info.samplerate)
        seen += 1
        if seen >= 30:
            break
print(f"\nsampled {seen} FLACs: sample-rates {dict(srs)}, "
      f"dur median {statistics.median(durs):.2f}s range {min(durs):.1f}-{max(durs):.1f}s")
