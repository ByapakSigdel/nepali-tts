# Parallel training (DiLoCo) — design & decision

**Status:** deferred to Stage B (not implemented on the Stage-A epoch-239→350 run). **Date:** 2026-06-09.

The user asked to train on the laptop GPU **and** a free Colab GPU *in parallel* (not the one-at-a-time
relay). We researched it (5-angle web study + adversarial verification) and decided **NOT to do it on
the current run**, but to keep this design for the large **Stage B (SLR54)** run, where it actually pays
off. This file preserves the design so Stage B can implement it.

## Why not on the current run (the honest verdict)

Both load-bearing claims were adversarially judged **false** for this run:

1. **It barely helps this run.** The relay *already pools both GPUs' compute-hours* (sequentially); true
   parallel only adds *concurrency*, capped at 2× for N=2 and eroded to a realistic **~1.3–1.6×** by sync
   overhead on the ~2–7 MB/s home uplink, Colab preemption (~60% availability), and the slow worker
   gating the barrier. And the run is essentially done: loss plateaued by ~epoch 21; 350 is an arbitrary
   stop. Racing the flat tail faster saves a few hours of near-zero-marginal-quality compute.
2. **Averaging a GAN is risky and currently unmeasurable.** No published work averages a live adversarial
   VITS/HiFi-GAN across machines. It would *probably* survive here (shared—not sharded—data; no
   BatchNorm; late-stage same-basin), but it could subtly degrade quality, and we had **no quality
   metric** to detect that (now partly fixed: see CER below). It also means swapping a hardened,
   55-agent-reviewed relay for unproven code.

**DiLoCo's real home is Stage B** (SLR54 ≈ 157k utts, a fresh long large-data run): full token budget to
amortize inner loops, enough data that an H-step inner loop is < 1 epoch, real shard diversity.

## The algorithm (DiLoCo, for when we do it)

Two-level optimization (DeepMind arXiv:2311.08105; Scaling-Laws arXiv:2503.09799; OpenDiLoCo
arXiv:2407.07852):

- **Inner:** each worker trains independently for **H** steps with the *existing* piper AdamW optimizers
  (keep piper's inner LR — do NOT use the paper's 4e-4). AdamW moment buffers are **local**, never synced.
- **Outer (the only new math):** every H steps, `pseudo_grad = θ_base − θ_after_H` (OLD minus NEW),
  averaged across present workers `Δ = mean(pseudo_grad_i)`, applied with **SGD + Nesterov, outer_lr=0.4
  (low end — M=2 is few replicas), momentum=0.9**. PyTorch form: `b←μb+g; g←g+μb; θ←θ−lr·g`. The outer
  momentum buffer **persists** across rounds (checkpoint it).
- **What to average:** **both `model_g` AND `model_d`** (incl. the speaker-embedding table
  `model_g.emb_g.weight`). Do *not* keep D local — with shared data a desynced D vs a merged G is what
  spikes the adversarial loss. Non-float buffers: take the anchor's values, don't run them through SGD.
- **H:** start at **~1 epoch** (small H fights GAN client drift); DyLU-scale Colab's H by throughput so
  both reach the barrier together.

### Confirmed model layout (from `scripts/_probe_ckpt.py` on a real ckpt — the design's assumed
`net_g`/`net_d` names were WRONG):
- Generator prefix = **`model_g.`** (693 float tensors). Discriminator = **`model_d.`** (111).
- Speaker embedding = **`model_g.emb_g.weight`**.
- Top-level ckpt keys: `epoch, global_step, state_dict, optimizer_states (2: G,D), loops, callbacks, ...`.

## Integration (no trainer fork)

**Primary = Approach B (segmented):** an orchestrator loops `python -m piper.train fit … --trainer.max_steps
<base+H> --ckpt_path last.ckpt` (exactly like `10_train_autoresume.sh`), then between segments loads
θ_after from the written ckpt, computes the delta vs a θ_base snapshot, does the HF rendezvous + average +
outer SGD, writes the merged `model_g`/`model_d` back into the ckpt's `state_dict`, and continues. Zero
changes to Lightning internals; survives the CUDA-fault auto-resume.
**Fallback = Approach A:** a Lightning `Callback` (same `on_train_batch_end` hook as `status_writer.py`)
that runs the outer round inline at H boundaries — smaller diff but does blocking network I/O inside the
training loop.

## Transport / rendezvous (reuse `hf_sync.py`)

All DiLoCo files under a **`diloco/` prefix** so they never collide with the relay's
`last.ckpt`/`PROGRESS.json`/`LOCK.json` — **structurally cannot corrupt the real run.**
- Ship **weights-only fp16 deltas (~70 MB)**, never the 882 MB checkpoint (10–35 s/round on the uplink).
- Repo layout: `diloco/master/{round.json, theta.safetensors, outer_state.safetensors}`,
  `diloco/rounds/<R>/{delta_<dev>.safetensors, ready_<dev>.json}`, `diloco/DILOCO_LOCK.json`.
- New `hf_sync` subcommands (thin wrappers on the existing CAS-commit/lock helpers): `post-delta`,
  `await-peer --timeout`, `get-master`, `put-master`.
- **Barrier with grace:** laptop = anchor+aggregator (never preempted); Colab = volatile. Anchor awaits
  the peer for a bounded grace window; if the peer is absent/stale → **proceed SOLO** (a 1-worker outer
  step is valid and cannot corrupt the master). First cut: single designated aggregator (laptop computes
  the new master; Colab just pulls it) — masterless lockstep is a later optimization.

## Guardrails (the #1 "no progress lost" rule)

- **Prerequisite (now partly done):** a real quality metric. Round-trip **CER** via `scripts/eval_cer.py`
  on a fixed sentence set; plus a held-out val split (pin the `random_split` seed; include the lone male
  speaker + data-poor speakers). Track `loss_d` (~2.7, must not diverge) at sync boundaries.
- **Fallback state machine:** every K rounds export ONNX → CER. Accept if `CER ≤ CER₀+ε` and `loss_d` ok;
  reject/rollback (halve outer_lr, retry once, else abort to the relay) after 2 bad checks.
- **The relay's `last.ckpt` is never written by DiLoCo;** promotion of a DiLoCo result to the real run is
  a deliberate, guarded `push-ckpt` (still subject to the epoch-overwrite guard, rc 9).

## Incremental rollout (prove before scaling)

1. **Math/format unit test** (no net/GPU): fp16 round-trip; sign/no-op SGD test (catches the sign flip);
   masterless determinism (same base+Δ → byte-identical).
2. **2-process rendezvous on the laptop alone** against a throwaway `diloco_test/` prefix: post/await/
   aggregate/put/pull → identical sha256; kill one mid-round → other proceeds solo, master stays valid.
3. **Single-GPU "fake parallel" smoke** (peer always absent) → segments clean, weights write back, CER runs.
4. **Add Colab as worker 2** on a *forked* epoch-checkpoint copy; ~10–20 rounds; watch `loss_d`+CER.
5. **Only if validated → use for Stage B.**

## The safe alternative we took instead (Stage A)

Keep the relay (already pools both GPUs' hours, provably no progress loss) + **generator model-soup / EMA
for export** (`scripts/snapshot_generator.py` + `scripts/soup_export.py`) — average late-plateau generator
checkpoints *for the exported voice only*, never fed back into training. Proven-safe, no distributed risk.
Plus the new CER metric for an objective quality number.
