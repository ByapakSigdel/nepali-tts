# Engineering Devlog — Nepali Multi-Speaker TTS

A chronological lab notebook of **every notable problem, its root cause, and the fix**. This is the
detailed companion to [`../PLAN.md`](../PLAN.md) (plan + status) and [`paper.tex`](paper.tex) (formal writeup).

**Entry format:** *Problem → Symptom → Root cause → Fix → Status.* Newest at the bottom. Append as we go.

---

## Phase 0 — Environment (2026-06-03)

**0.1 GPU usability on a brand-new Blackwell laptop GPU.**
- *Symptom:* RTX 5050 Laptop is `sm_120` (Blackwell); most PyTorch builds only support up to `sm_90`.
- *Root cause:* Stable PyTorch wheels lag new GPU architectures.
- *Fix:* Install PyTorch **cu128** wheels (`--index-url https://download.pytorch.org/whl/cu128`) → torch
  2.11.0+cu128, which has `sm_120` kernels. Verified: `torch.cuda.get_device_capability` = (12,0), GPU matmul OK.
- *Status:* ✅ Resolved.

**0.2 Linux tooling on Windows.**
- *Fix:* Use WSL2 **Ubuntu-24.04** (Python 3.12). GPU is visible inside WSL via the Windows driver. Keep code on
  `/mnt/c`, but data/venv/checkpoints on the WSL-native fs (`~/voicemodel`) for I/O speed.
- *Status:* ✅. (sudo needs a password → system-package installs are a manual user step.)

---

## Phase 1–3 — Data & baseline (2026-06-08)

**1.1 OpenSLR dataset identities were not what their numbers implied.**
- *Symptom:* Assumed SLR43 = big ASR set.
- *Root cause:* Misremembered. Verified from openslr.org: **SLR43** = multi-speaker female TTS (800 MB),
  **SLR54** = large ASR ~157k utts (~9 GB), **SLR143** = male+female TTS (165 MB).
- *Fix:* Staged plan — Stage A = SLR43+SLR143 (TTS-grade, ~965 MB); Stage B = SLR54 later.
- *Status:* ✅.

**1.2 Download script exited 141 ("failed") but data was fine.**
- *Root cause:* A trailing `find … | head` tripped `set -o pipefail` (SIGPIPE) *after* all real work succeeded.
- *Fix:* Recognize cosmetic pipefail; data verified intact.
- *Status:* ✅.

**1.3 torchaudio.load broken in torch 2.11.**
- *Symptom:* `TorchCodec is required for load_with_torchcodec`.
- *Root cause:* torchaudio 2.11 moved I/O to a separate TorchCodec package.
- *Fix:* Load with **soundfile** instead; resample with `torchaudio.functional.resample` (math only, no codec).
- *Status:* ✅.

**3.1 MMS baseline model id + input format.**
- *Symptom:* `facebook/mms-tts-npi` → 401; then Devanagari → empty tokens → crash.
- *Root cause:* Correct id is **`mms-tts-npl`**; its tokenizer vocab is **romanized Latin** (its `is_uroman`
  flag wrongly reports False).
- *Fix:* Use `npl`; romanize Devanagari with **uroman** before tokenizing.
- *Status:* ✅ (baseline audio generated).

---

## Phase 4 — The Blackwell training saga (2026-06-08)

The hard part. VITS training on `sm_120` + WSL2 + torch 2.11 hit a chain of undocumented failures.

**4.1 TorchScript fused-op crash.**
- *Symptom:* `RuntimeError … TorchScript interpreter … CUDA out of memory. Tried to allocate 20 MiB`
  (with ~6 GB free — i.e. not a real OOM), inside `fused_add_tanh_sigmoid_multiply`.
- *Root cause:* VITS decorates that op with `@torch.jit.script`; the JIT kernel fails on `sm_120`, surfaced
  as a bogus OOM.
- *Fix:* `export PYTORCH_JIT=0` (run the op eagerly).
- *Status:* ✅ (got past it; revealed 4.2).

**4.2 `expandable_segments` spurious OOM under WSL.**
- *Symptom:* After adding `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, OOM with
  "this process has 17179869184.00 GiB memory in use" (16 EiB — garbage).
- *Root cause:* `expandable_segments` uses CUDA virtual-memory APIs that are broken under WSL2; the bogus
  accounting line appeared **only** with that flag set.
- *Fix:* **Do not** set `expandable_segments`. (This was self-inflicted by my "memory hygiene" over-correction.)
- *Status:* ✅.

**4.3 Precision: bf16 and fp16 both unusable.**
- *Symptom:* `bf16-mixed` → `RuntimeError: cuFFT doesn't support tensor of type: BFloat16`. `16-mixed` → trained
  ~16 steps then a CUDA fault.
- *Root cause:* cuFFT has no bf16 path (mel STFT); fp16 uses an experimental "ComplexHalf" FFT that is unstable
  on Blackwell.
- *Fix:* Use **`32-true` (fp32)** — fully supported, no experimental paths.
- *Status:* ✅.

**4.4 Intermittent backward-pass `CUDA error: unknown error`.** *(the big one)*
- *Symptom:* With fp32 it trained much further (step 163, then 304) but still crashed at *random* steps inside
  `loss.backward()`. Non-deterministic → not a config/code bug. Worse at higher batch (batch8→step16,
  batch4→step163/304).
- *Root cause:* A **known Blackwell `sm_120` driver fault** in the backward pass (corroborated by an NVIDIA dev
  forum report).
- *Fixes (layered):* (a) NVIDIA driver **596.36 → 610.47** roughly halved the crash rate; (b) batch size 2;
  (c) an **auto-resume loop** (`scripts/10_train_autoresume.sh`) that checkpoints every 100 steps and restarts
  from `last.ckpt` after any fault → training is unattended and fault-tolerant.
- *Tried and rejected:* `CUDA_LAUNCH_BLOCKING=1` made it stable but **~50× too slow** (~1 step/min) — unusable.
- *Status:* 🟡 Mitigated, not eliminated. Training progresses reliably via auto-resume.

**4.5 PowerShell wrapper OOM on big error dumps; WSL daemon got reaped.**
- *Root cause:* Piping huge tracebacks through PowerShell `Tee-Object` exhausted the PS process; and a
  `setsid nohup` daemon launched via `wsl -- …` is torn down when the launching `wsl` command returns.
- *Fix:* Redirect logs **inside** WSL (`> log 2>&1`); run the loop as a harness-tracked background task (keeps
  a live `wsl` process holding the session open).
- *Status:* ✅.

**4.6 WSL wedged after repeated GPU faults; `wsl --shutdown` hung.**
- *Symptom:* All `wsl` commands returned empty; `wsl --shutdown` hung. Host `nvidia-smi` worked (GPU idle).
- *Root cause:* Repeated CUDA faults left the GPU/WSL VM in a stuck state.
- *Fix:* **Full Windows reboot** (clean slate; also applied new `.wslconfig` resource caps + the new driver).
- *Status:* ✅.

**4.7 Training "stuck at epoch 0" / GPU looked idle.**
- *Symptom:* After reboot, tracker showed 0 for ~28 min; no checkpoint saved; GPU ~16% util.
- *Root causes:* (a) Windows Task Manager shows the **"3D" engine** by default — WSL CUDA load is on the
  **"Cuda"/"Compute" engine** (a different, hidden graph), so it *looked* idle; (b) genuinely ~20× too slow,
  starved for data.
- *Fix for (b):* **`--data.num_workers 4`** (feed the GPU) and removed the `nice -n 10` priority handicap;
  kept WSL RAM/core caps (`.wslconfig`) for host responsiveness. Result: real progress, loss dropping.
- *Status:* ✅.

**4.8 No live progress in the log (tracker stuck on "starting up").**
- *Symptom:* Lightning's progress bar writes nothing to a redirected (non-TTY) log → no epoch/step visible.
- *Fix:* A small Lightning callback **`scripts/status_writer.py`** writes `epoch/step/loss` to `status.txt`
  every 10 steps; `scripts/progress.sh` reads it. (Plus a parser bug: `grep 'epoch='` also matched
  `step_in_epoch=` → anchored it to `(^| )epoch=`.)
- *Status:* ✅ (live tracker works).

---

## Phase — Inference / sampling (2026-06-08)

**5.1 ONNX export — missing dep then exporter failure.**
- *Symptom:* `ModuleNotFoundError: onnxscript`; then the torch-2.11 **dynamo** exporter failed on VITS spline
  flows (`rational_quadratic_spline` discriminant assertion).
- *Fix:* `pip install onnxscript`; patch `export_onnx.py` to use the **legacy exporter** (`dynamo=False`).
  (Also: don't set `PYTORCH_JIT=0` for export — the legacy exporter needs JIT tracing; the JIT issue was
  GPU-training-only and we export on CPU.)
- *Status:* ✅ (77 MB ONNX exported).

**5.2 piper inference ignored `-c` and looked for `<model>.onnx.json`.**
- *Fix:* Copy the training `config.json` to `ne.onnx.json` next to the model.
- *Status:* ✅ (first Nepali audio synthesized across speakers).

---

## Phase — Documentation (2026-06-08)

**6.1 Fragmented training logs across 20 versions.**
- *Root cause:* Every restart created a new `lightning_logs/version_N`.
- *Fix:* `scripts/plot_training.py` (tbparse) merges all tfevents by step → `docs/*.png`. Result so far:
  `loss_g` 47→31 (descending), `loss_d` ~2.7 (stable).
- *Status:* ✅.

**6.2 Research paper.** Markdown ([`PAPER.md`](PAPER.md)) + LaTeX ([`paper.tex`](paper.tex), IEEEtran,
pdfLaTeX-safe, romanized for portability). CER eval harness `scripts/eval_cer.py` prepared (run once trained).
- *Status:* ✅ (Overleaf-ready bundle).

**6.3 Expanded figure set** (figures matter for a paper, so use them where data allows). Added via
`scripts/make_figures.py`, from real data: a **system pipeline** diagram, a ground-truth-vs-synthesized
**mel-spectrogram** comparison of the same utterance (shows the model recovers harmonics/formants), and a
**clip-duration histogram**. `scripts/bundle_paper.ps1` now auto-bundles every `docs/*.png` so new figures are
never left out of the Overleaf zip.
- *Status:* ✅.

---

## Phase — Frontend: offline web app (2026-06-08)

**7.1 User-facing TTS app.**
- *Goal:* type Nepali $\rightarrow$ choose a voice $\rightarrow$ get audio, fully offline.
- *Build:* **Gradio** web app (`frontend/app.py`): text box, speaker dropdown (reads `speaker_id_map`
  from the model config), speed/variation sliders, auto-playing audio output. Synthesis shells out to the
  proven `python -m piper`. Launcher `frontend/run.sh` re-exports the latest checkpoint (best-effort, with
  fallback) then serves on `0.0.0.0:7860`.
- *Access:* reachable from the Windows browser at `http://localhost:7860` via WSL `localhostForwarding`
  (set in `.wslconfig`). Verified `HTTP 200`.
- *Gotcha:* Gradio 6.0 moved the `theme=` argument from `gr.Blocks(...)` to `demo.launch(...)` (warning only);
  fixed.
- *Status:* ✅ Live.

**7.2 Romanized + code-mixed input.**
- *Goal:* let users type informal romanized Nepali (Latin) and code-mixed English — e.g. "hello sanchai
  hunuhuncha tapai" — instead of Devanagari, since the model needs Devanagari.
- *Decision (via a 3-agent research workflow):* use **`indic-transliteration` (sanscript, OPTITRANS scheme)** —
  pure-Python (deps: regex/typer/toml/roman/tqdm), so it does **not** touch torch 2.11. *Rejected* ai4bharat
  IndicXlit: hard `fairseq`+`tensorflow` deps, fails on Python 3.12, and would resolve a conflicting torch.
- *Build:* `frontend/translit.py` = a curated common-word dictionary (high accuracy on frequent words) + an
  OPTITRANS rule fallback + a Nepali phonetic normalizer; Devanagari/punctuation/digits pass through. v1
  code-mixing = single-script normalization (English routed through the same engine → Nepali-accented, keeps
  espeak-ne in one language). The app gained an **editable Devanagari preview** as the safety net for the
  ~80% deterministic-transliteration ceiling.
- *Verified:* "kasto cha tapaiko aaile" → कस्तो छ तपाईंको अहिले; "hello sanchai hunuhuncha tapai" →
  हेलो सञ्चै हुनुहुन्छ तपाईं; torch still 2.11.0+cu128.
- *Status:* ✅ Live (v1). A genuinely natural, casually code-switched tone remains future work (needs
  code-mixed Nepali-English data).

**7.3 Live model refresh.** The served voice could go stale (the app bakes in the checkpoint only at launch,
and `REFRESH=0` skips even that). Added a **"🔄 Update to latest training checkpoint"** button that re-exports
`last.ckpt` → the served ONNX *live* (no restart); the app also ensures a model exists at startup (export
latest, else fall back to the prior sample). Users can now pull in newer training mid-session.
- *Status:* ✅.

## Phase — Throughput tuning + ETA (2026-06-08)

**8.1 Full-usage training** (user away from the laptop → maximize speed). Raised WSL caps
(`.wslconfig` 8GB/8c → 12GB/14c, applied via `wsl --shutdown`), bumped `--data.batch_size` 2→4 and
`--data.num_workers` 4→8. Result: GPU util ~50% → ~75%, throughput ~**2.85×** (2.86 it/s at batch 4 vs ~2 it/s
at batch 2), **0 crashes**, and ~2.3 GB VRAM still free (room to push batch higher). Auto-resume still guards
the intermittent Blackwell fault. Note: VITS trains on fixed-size segments, so VRAM scales weakly with batch
(5.2 GB @ b2 → 5.6 GB @ b4) — larger batches are cheap.

**8.2 ETA in the tracker.** `scripts/progress.sh` now estimates the rate from two reads of `status.txt`
(fractional epochs/sec, via awk) and shows **it/s, epochs/hour, time-to-next-epoch, and ETA to a target epoch**
(`TARGET=N`, default 350).
- *Status:* ✅.

**8.3 Training target = 350 epochs.** Set `--trainer.max_epochs 350` (was a 2000 placeholder cap) so the
auto-resume loop **stops itself at epoch 350** — a defensible endpoint (~19 h from epoch ~27 at ~16.7
epochs/hr; the loss already plateaued by ~21). `progress.sh` ETA target defaults to 350 to match.
- *Status:* ✅.

## Phase — Test at epoch 221 + light mode (2026-06-09)

**9.1 Training silently stalled at epoch 221.** Found training *not running* while we believed it was
climbing to 350. Root cause: an earlier background **relaunch had failed** — the relaunch command passed a
`/mnt/c/...` path as a top-level argument through Git Bash, whose MSYS layer rewrote it to
`C:/Program Files/Git/mnt/c/...` (the same path-mangling that bit the frontend launch). The auto-resume loop
never came back up, so the model sat idle at 221. *Fix:* always wrap WSL paths **inside** the quoted
`bash -lc '...'` script body (never as a bare arg) so MSYS leaves them alone; relaunched and confirmed it
resumed from `last.ckpt`. *Lesson logged:* prefer `wsl -d Ubuntu-24.04 -- bash -lc 'exec bash /mnt/c/.../x.sh'`.
- *Status:* ✅ resumed from epoch 221.

**9.2 Tested epoch-221 voice via the web app.** Exported the epoch-221 checkpoint to ONNX and served it
(`frontend/run.sh`, REFRESH=1). Clearly more mature than the epoch-66 version. CPU-only inference, so it runs
alongside GPU training without contention.

**9.3 Switched to LIGHT mode** (user is on the laptop, wants it responsive). Reverted `.wslconfig`
12GB/14c → **8GB/8c** (`wsl --shutdown` to apply), and lowered `--data.batch_size` 4→**2**,
`--data.num_workers` 8→**4**. Result: GPU ~75% → **~50%** (cooler/quieter), 8 cores freed for Windows.
*Trade-off documented for the user:* epoch-rate dropped ~16.7 → **~4.2 epochs/hr** (ETA to 350: ~13 h → **~30 h**).
Two compounding causes: fewer cores/workers (lower it/s) **and** batch-size 2 doubling steps/epoch (616 → 1231
batches). Toggle back to full-usage when away. Both `.wslconfig` and `10_train_autoresume.sh` carry comments
explaining the two modes.
- *Status:* ✅ training resumed at epoch 224 toward 350.

## Open / ongoing
- Training accumulating (auto-resume); refresh graphs + run `eval_cer.py` as it matures.
- Stage B (SLR54) scale-up; Nepali correction lexicon (nasalization, loanwords, numbers, currency).
- Upload trained model to a GitHub Release; emotions = future phase (needs emotion-labeled data — none public for Nepali).
