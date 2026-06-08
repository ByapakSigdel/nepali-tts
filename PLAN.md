# Nepali Multi-Speaker TTS — Project Plan

> Goal: An **extremely accurate** Nepali text-to-speech model with **multiple selectable voices**,
> trained on **public datasets**, that runs **offline on a PC/laptop**.
> Constraint: training on a single **RTX 5050 Laptop GPU (8GB VRAM, Blackwell/sm_120)**.

Status legend: ⬜ not started · 🟨 in progress · ✅ done

---

## 0. Decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Voice target | Multi-speaker (pick voice by speaker ID) | User requirement |
| Data | Public Nepali corpora | User requirement |
| Deployment | Offline on PC → **ONNX** runtime | User requirement |
| Compute | Laptop only (8GB) → **warm-start, small batch** | User requirement |
| **Framework** | **Piper (VITS)** | Offline-first ONNX, multi-speaker, lightweight, espeak-ng Nepali G2P |
| Backup framework | Coqui TTS (`coqui-tts` fork) multi-speaker VITS | If Piper Nepali phonemes underperform |
| G2P | espeak-ng `ne` + custom normalization + correction lexicon | Schwa deletion is the #1 Nepali TTS failure |
| Training OS | WSL2 Ubuntu (CUDA via WSL) | Piper/Coqui pipelines are Linux-first |
| Sample rate | 22,050 Hz mono | Piper/VITS standard |

### Honest expectations
- Multi-speaker + crowdsourced public audio = solid & intelligible, **not studio-perfect** per voice.
  Option to add one **single clean speaker "premium" voice** later for a standout.
- "Extremely accurate" is won in the **text frontend** (normalization + G2P), not just the model.

---

## 1. Datasets (verified from openslr.org)

| ID | Content (real) | License | Size | Stage |
|---|---|---|---|---|
| **SLR43** | High-quality **multi-speaker TTS**, female (ne-NP); `line_index.tsv`; speaker ID prepended to each filename | CC BY-SA 4.0 (commercial OK) | 800 MB | **A** (primary) |
| **SLR143** | Nepali **TTS**, **male + female**; separate male/female `.tsv` (audio_id + sentence) | CC BY-**NC**-SA 4.0 (non-commercial) | 165 MB | **A** (adds male) |
| **SLR54** | Large Nepali **ASR**, **~157K utts**, many speakers; `utt_spk_text.tsv` | CC BY-SA 4.0 | ~17 GB | **B** (scale-up) |
| Common Voice `ne` | Crowd read speech | CC0 | — | optional |
| `facebook/mms-tts-npl` | Pretrained Nepali VITS (single-spk, romanized) | CC BY-NC | — | baseline only (done) — NOT for warm-start |

**Staged approach:** Stage A = SLR43 + SLR143 (~965 MB, TTS-grade, female+male) → build pipeline + first model.
Stage B = add SLR54 (17 GB ASR) only if we need more speaker variety.
**License flag:** SLR143 & MMS are non-commercial; SLR43 & SLR54 are commercial-OK. Matters only if this ever ships commercially.
Mirrors: `openslr.elda.org` and `openslr.trmal.net` work; `openslr.magicdatatech.com` is dead.

Per-speaker selection criteria for TTS: enough clean utterances/speaker, consistent recording, low noise.

---

## 2. Pipeline phases

### Phase 0 — Environment ✅
- [x] Verify/enable **WSL2 + Ubuntu** — ✅ WSL 2.6.3, **Ubuntu-24.04** chosen (Python 3.12, gcc 13.3)
- [x] Confirm GPU visible in WSL — ✅ RTX 5050 8GB seen via `nvidia-smi`, WSL CUDA libs present
- [x] **system deps** (sudo) — ✅ ffmpeg, espeak-ng 1.51 (+ `ne` voice), libsndfile1 installed
- [x] Python venv at `~/voicemodel/.venv`; **PyTorch cu128** — ✅ torch 2.11.0+cu128
- [x] GPU smoke test — ✅ CUDA True, **sm_120**, GPU matmul OK

**Environment facts (confirmed):**
- Host: AMD Ryzen 7 260 (16 threads), 15.3 GB RAM, RTX 5050 Laptop 8GB (Blackwell sm_120), driver 596.36
- WSL: Ubuntu-24.04, default RAM 7.4GB → raised to 11GB via `C:\Users\user\.wslconfig` (apply on next `wsl --shutdown`)
- **Paths**: code/scripts/plan live in Windows `c:\Users\user\Documents\VoiceModel` (= `/mnt/c/...` in WSL);
  large data + venv + checkpoints live in WSL-native `~/voicemodel/` for I/O speed
- sudo requires password (not passwordless) → system installs are a manual user step

### Phase 1 — Data acquisition 🟨
- [x] Verify datasets, sizes, licenses, mirrors (see §1)
- [x] **Stage A download** ✅ SLR43 + SLR143 → `~/voicemodel/data/raw/`
- [x] Inventory ✅ (see below)
- [ ] (Stage B) SLR54 17 GB — decision pending

**Stage A inventory (measured):**
- **SLR43**: 2,064 clips / 2.80 h / **18 female speakers** / 48 kHz mono. IDs like `nep_0546`.
  Heavily imbalanced: utts/speaker min 5, median 61, **max 505** (`nep_0546`).
- **SLR143**: 675 clips / 1.24 h / **1 female (566) + 1 male (109)** / 22.05 kHz mono. tsv has header row `audio_id\tsentence`.
- **Combined: 2,739 clips, 4.04 h, ~20 speakers.** Mostly female; only 1 male speaker.
- Implications: resample all → 22.05 kHz; expect strong voices only for data-rich speakers; male voice weak until SLR54.

### Phase 2 — Data preparation 🟨
- [x] Resample → 22.05kHz mono; peak-normalize; trim silence ✅ (`06_preprocess_stageA.py`)
- [x] Build multi-speaker manifest `wav|speaker|text` ✅ `~/voicemodel/data/processed/metadata.csv`
      (2,739 clips, **3.83 h**, 20 speakers; strongest: slr143_female 59m, nep_0546 35m, nep_2099 29m)
- [ ] **Text normalization** (numbers, dates, currency रू, English loanwords, punctuation) — refine later
- [ ] Phonemize via espeak-ng `ne` — happens in Piper preprocess (Phase 4)
- [ ] Train/val split (held-out per speaker) — happens in Piper preprocess (Phase 4)

### Phase 3 — Baseline 🟨
- [x] Run **`facebook/mms-tts-npl`** on test set → ✅ reference audio in `eval/mms/`
      (notes: needs **uroman** romanization first — its `is_uroman` flag is wrong; output 16kHz, **single speaker**, can't speak digits)
- [ ] Establish objective baseline (ASR round-trip CER) — deferred until eval harness (Phase 5)

### Phase 4 — Training 🟨
- [x] Trainer = **piper1-gpl** (`pip install -e .[train]`, torch 2.11 kept); espeak `ne`; metadata.csv = our format
- [x] Warm-start = `--model.vocoder_warmstart_ckpt` lessac-medium (single→multi needs non-strict vocoder load)
- [x] First checkpoint saved (`models/ne_stageA/ckpts/last.ckpt`, 924MB)
- [ ] Accumulate training steps to intelligible Nepali (auto-resume daemon running)
- [ ] Export ONNX + sample

**⚠️ Blackwell (sm_120) + WSL2 + torch 2.11 training gotchas (hard-won):**
1. `@torch.jit.script` fused op crashes in TorchScript interpreter → **`export PYTORCH_JIT=0`** (run eager).
2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → CUDA VMM broken under WSL → bogus "16 billion GB" OOM. **Do NOT set it.**
3. `bf16-mixed` → cuFFT can't STFT bf16. `16-mixed` → experimental ComplexHalf FFT unstable. **Use `32-true` (fp32).**
4. **Intermittent `CUDA error: unknown error` in `loss.backward()`** at random steps (16, 163, ...) — driver/HW fault under load,
   worse at higher batch. Mitigation: **batch 2**, checkpoint every 25 steps, **auto-resume loop** (`scripts/10_train_autoresume.sh`),
   run as **detached WSL daemon** (`setsid nohup`) so it survives the flaky PowerShell wrapper.
5. Log INSIDE wsl (`> log 2>&1`), not via PowerShell `Tee-Object` (PS process OOMs on big error dumps).
TODO to try if still unstable: update NVIDIA driver; `CUDA_LAUNCH_BLOCKING=1` (slower, may stabilize).

### Phase 5 — Evaluation ⬜
- [ ] **Objective**: ASR round-trip CER/WER (MMS-ASR or Whisper-large-v3) on generated audio
- [ ] **Subjective**: listening MOS on fixed sentences
- [ ] **Hard-word set**: schwa-deletion words, conjuncts, loanwords, numbers, dates
- [ ] Scorecard vs MMS baseline

### Phase 6 — Iterate ⬜
- [ ] Build/extend **correction lexicon** for mispronounced words
- [ ] Improve normalization from failures
- [ ] Add/curate speakers; retrain

### Phase 7 — Ship ⬜
- [ ] Export **ONNX** voices
- [ ] Offline inference CLI (text → wav, choose speaker)
- [ ] Optional simple GUI
- [ ] Package + usage docs

---

## 3. Repo layout
```
VoiceModel/
├── PLAN.md            # this file (living doc)
├── data/              # raw + processed corpora (gitignored, large)
├── scripts/           # download, prep, normalize, train, eval scripts
├── frontend/          # Nepali text normalization + G2P + lexicon
├── configs/           # training configs
├── models/            # checkpoints + exported ONNX
├── eval/              # test sets, hard-word lists, scorecards
└── notes/             # research notes, decisions log
```

## 4. Open risks
- Crowdsourced audio cleanliness limits per-voice quality → curate hard.
- espeak-ng Nepali schwa accuracy → measure, patch with lexicon.
- Windows/Blackwell toolchain friction → WSL2 + cu128 mitigates.
- 8GB VRAM → warm-start + small batch + grad accumulation mandatory.
