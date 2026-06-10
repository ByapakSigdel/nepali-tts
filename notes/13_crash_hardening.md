# 13 — Stage B crash storm: root cause + hardening (2026-06-10)

## Symptom
Stage B training crash-looped: 14+ auto-resume restarts in one day. Every crash was the
same `EOFError` from `torch.load` on a cache `.pt` file, at one of two sites:

- `dataset.py` `prepare_data` (line ~355): loading a cached `audio.pt` to compute a missing spectrogram
- `dataset.py` `VitsDataset.__getitem__` (line ~588): loading a cache file for a training batch

Alongside: WSL itself failed twice at the service level ("Catastrophic failure",
`Wsl/Service/CreateInstance/E_FAIL`, transient `0x8007274c` connection errors), and one
user hard-shutdown while caching was in flight.

## Root cause
A vicious cycle of **non-atomic cache writes** meeting **unclean terminations**:

1. piper writes each cache file with a plain `torch.save(path)` — a crash/kill mid-write
   leaves a **truncated file** at the final path.
2. piper's cache check is `path.exists()` — a truncated file looks "done" and is never
   rewritten. It is a permanent landmine.
3. Random batch sampling eventually loads the landmine → `EOFError` → the whole run dies.
4. The auto-resume loop restarts the trainer… whose death (or the next WSL crash) kills
   8 dataloader workers mid-write, **minting new truncated files**. GOTO 3.

Scale of damage measured: after the WSL crash + one hard shutdown, a full-cache scan
(`scripts/clean_cache.py`) found **1,983 corrupt files across 662 clips (~0.75%), scattered
uniformly through the cache** — classic lost-writeback-page corruption, not just the
files open at the instant of death.

## Fix (patched into the piper checkout, exported as `patches/piper_local.patch`)
Two complementary changes in `src/piper/train/vits/dataset.py`:

1. **Atomic cache writes** — every cache `torch.save` now writes to `<file>.tmp` then
   `os.replace`s it into place. A truncated file at the final path is now *impossible*;
   the worst case is a harmless `.tmp` orphan (cleaned by `scripts/stop_stageB.sh`).
2. **Self-healing reads** — both load sites now catch corruption instead of dying:
   - `prepare_data`: a corrupt `audio.pt` is deleted and **regenerated from the wav**
     (logic factored into `_compute_norm_audio()`).
   - `__getitem__`: the corrupt file(s) are deleted (regenerated on the next
     `prepare_data`) and the **next utterance is substituted** for the batch, so
     training continues uninterrupted.

Supporting hardening from the same incident:
- `scripts/ckpt_guardian.py` — rotating, validated, disk-synced backups of `last.ckpt`
  (the same truncation mechanism could hit the checkpoint and lose days of progress).
- `scripts/pick_ckpt.py` — the resume loop now validates `last.ckpt` and falls back to
  the newest valid backup instead of crash-looping on a truncated checkpoint.
- `scripts/clean_cache.py` — full-cache scanner/sweeper (one-off recovery tool).
- `scripts/stop_stageB.sh` — clean stop of loop + trainer + guardian, sweeps `.tmp` orphans.
- `colab/colab_train.ipynb` cell 3 now applies `patches/piper_local.patch` after cloning
  stock piper, so Colab runs the identical hardened code (it also inherits the smart
  warm-start and the onnx `dynamo=False` fix it previously lacked).

## Operational lessons (Windows/WSL specifics)
- **wsl.exe mangles inline quoting**: double-quoted patterns inside `bash -lc '…'` lose
  their quotes crossing the Windows→WSL boundary (commands like `grep -c "A B"` silently
  break, `pkill -f` can self-match and kill its own shell). Rule: put any non-trivial
  shell logic in a `.sh` file under `scripts/` and run `wsl bash /mnt/c/...` (also dodge
  Git-Bash path mangling by invoking wsl from PowerShell, not the Bash tool).
- **`wsl --shutdown` is the only way to give the RAM back**: after heavy I/O, the WSL VM
  (vmmem) keeps the Linux page cache; stopping the processes is not enough.
- The cache survives WSL crashes/shutdowns fine *now that writes are atomic*; data on the
  ext4 vhdx was never the problem — in-flight writes were.

## State at time of writing
- Local Stage B stopped deliberately (laptop needed); WSL shut down; RAM freed.
- Cache: ~127k/128,009 clips done (~95 GB) — survives shutdown, finishes in minutes on next start.
- Checkpoint: `~/voicemodel/models/ne_stageB/ckpts/last.ckpt` (epoch 0, step ~800, batch 4).
- Next: take the Stage-B run to Colab via the checkpoint relay. **Still required for that**:
  upload Stage-B data + checkpoint + a Stage-B yaml to the HF repos and parameterize the
  notebook (it is currently hardwired to Stage A: `ne_stageA`, `num_speakers 20`,
  `processed.tar.gz`, `TARGET=350`).
