# Nepali Multi-Speaker TTS

An **offline, multi-speaker Nepali (Devanagari) text-to-speech** model — trained with
[piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) (VITS) on public OpenSLR data, exportable to ONNX
to run locally with no internet.

> **This repo holds the code, scripts, configs, and full reproduction recipe.**
> The big artifacts (datasets, the Python venv, and trained checkpoints) are **not** committed —
> they are re-downloaded / rebuilt on any machine by the steps below. See **"Saving the model"** for the checkpoint.

See [`PLAN.md`](PLAN.md) for the full design, decisions, dataset details, and the hard-won
**Blackwell/WSL2 training gotchas** (read that before training on an RTX 50-series GPU).

---

## What's where

| Lives in git (this repo) | Lives outside (rebuilt on each machine) |
|---|---|
| `scripts/` `configs/` `frontend/` | dataset audio → `~/voicemodel/data/` (scripts 03, 05, 06) |
| `PLAN.md` `README.md` `notes/*.md` | python venv → `~/voicemodel/.venv` (scripts 01, 07) |
| `eval/test_sentences_ne.txt` | warm-start ckpt → `~/voicemodel/checkpoints/` (script 08) |
| | trained model → `~/voicemodel/models/ne_stageA/` (training) |

**Path convention:** code lives in the Windows/clone folder; large data + venv + checkpoints live in the
**WSL-native** home (`~/voicemodel/`) for I/O speed.

---

## Reproduce on a brand-new machine

> Assumes Windows + WSL2 **Ubuntu-24.04** with an NVIDIA GPU. For an RTX 50-series (Blackwell) GPU,
> install **NVIDIA driver ≥ 610.47** on Windows first (older drivers cause training crashes — see PLAN.md).
> On a plain Linux box with an NVIDIA GPU, skip the WSL notes and run the scripts directly.

```bash
# 0. Clone this repo, then open WSL Ubuntu-24.04.
#    NOTE: scripts reference the path /mnt/c/Users/user/Documents/VoiceModel — if you cloned
#    somewhere else, find/replace that path in scripts/ to match your clone location.

# 1. System packages (ffmpeg, espeak-ng, python venv, build tools)   [needs sudo]
bash scripts/00_system_setup.sh

# 2. Python venv + PyTorch (cu128 build for Blackwell/sm_120)
bash scripts/01_python_env.sh

# 3. (optional) MMS Nepali baseline to compare against
bash scripts/02_mms_baseline.sh

# 4. Install the piper1-gpl trainer (keeps torch 2.11)
bash scripts/07_install_piper.sh

# 5. Download datasets
bash scripts/03_download_data.sh      # Stage A: SLR43 + SLR143 (~965 MB, TTS-grade)
bash scripts/05_download_slr54.sh     # Stage B: SLR54 (~9 GB ASR) — optional, more speakers

# 6. Preprocess Stage A -> 22.05kHz mono + manifest
~/voicemodel/.venv/bin/python scripts/06_preprocess_stageA.py

# 7. Get the warm-start checkpoint (lessac-medium, 807 MB)
bash scripts/08_get_lessac.sh

# 8. Train (auto-resumes through GPU faults; gentle on the laptop)
bash scripts/10_train_autoresume.sh > notes/10_autoresume.log 2>&1

# 9. Track progress anytime (epochs done, speed, GPU):
bash scripts/progress.sh
#    live view:  watch -n 10 bash scripts/progress.sh
```

---

## Saving / moving the trained model

The trained checkpoint (`~/voicemodel/models/ne_stageA/ckpts/last.ckpt`, ~900 MB) is **too big for a normal
git commit**. To carry it to a new machine, pick one:

- **GitHub Release asset** (recommended, up to 2 GB, doesn't bloat the repo):
  ```bash
  # after installing gh and `gh auth login`:
  gh release create model-ckpt-vN --title "checkpoint vN" --notes "training step N"
  gh release upload model-ckpt-vN ~/voicemodel/models/ne_stageA/ckpts/last.ckpt
  # on the new machine: download it back into ~/voicemodel/models/ne_stageA/ckpts/last.ckpt
  ```
- **Hugging Face Hub** (`huggingface-cli upload <repo> last.ckpt`) — best for model files.
- **Git LFS** (`git lfs track "*.ckpt"`) — works but the free tier has tight storage/bandwidth.

Once trained, export a runnable **ONNX** voice (this is the offline product):
```bash
~/voicemodel/.venv/bin/python -m piper.train.export_onnx \
  --checkpoint ~/voicemodel/models/ne_stageA/ckpts/last.ckpt \
  --output-file ne_stageA.onnx
```

---

## Status (2026-06-08)

Pipeline complete and working; first training in progress. Blackwell+WSL has an intermittent backward-pass
CUDA fault (mitigated by driver 610.47 + the auto-resume loop). Emotions/expressive speech is a **future
phase** (needs emotion-labeled data, which no public Nepali corpus has). Full detail in `PLAN.md`.
