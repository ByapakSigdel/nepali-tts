# Distributed training: home laptop + free Colab GPUs (checkpoint relay)

We pool a consumer laptop (Blackwell RTX 5050) and free Google Colab GPUs into **one** training
run. Only one machine trains at a time, but they **pass the baton** — the full training checkpoint —
through a private Hugging Face repo, so you get the *combined* GPU-hours of both. A lock stops them
from ever training at once, and full-state checkpoints mean **no progress is ever lost**.

```
   laptop trains ──push──▶  HF repo (last.ckpt = the baton)  ◀──pull── Colab trains
        ▲                         │  LOCK.json = whose turn          │
        └─────────────────────────┴──────────────────────────────────┘
```

## Pieces

| File | Role |
|---|---|
| `scripts/hf_sync.py` | the engine: `push`/`pull` the checkpoint, `claim`/`heartbeat`/`release` the lock |
| `scripts/11_train_relay.sh` | laptop wrapper: claim → sync → train → push → release |
| `colab/colab_train.ipynb` | Colab notebook: same flow, in the cloud |
| HF repo `…/nepali-tts-ckpt` | private — holds `last.ckpt`, `config.json`, `PROGRESS.json`, `LOCK.json` |
| HF repo `…/nepali-tts-data` | private — holds `processed.tar.gz` (training audio) |

Credentials live **outside** git, in `~/voicemodel/.hf_token` and `~/voicemodel/.hf_repo`.

## Run a Colab shift (your steps)

1. Open `colab/colab_train.ipynb` in Colab (File ▸ Open ▸ GitHub ▸ `ByapakSigdel/nepali-tts`).
2. **Runtime ▸ Change runtime type ▸ GPU.**
3. Add your HF token as a Colab **secret** named `HF_TOKEN` (🔑 icon, left sidebar).
4. **Runtime ▸ Run all.** It pulls the latest checkpoint, trains, and auto-pushes every ~20 min.
5. Leave the tab open. When you're done (or Colab disconnects), it has already pushed + released.

> **Important:** while Colab trains, the laptop must NOT also train (the lock enforces this — the
> laptop's `11_train_relay.sh` will wait). When you want the laptop to take over again, just start it.

## Hand the baton back to the laptop

```bash
wsl -d Ubuntu-24.04 -- bash -lc 'bash /mnt/c/Users/user/Documents/VoiceModel/scripts/11_train_relay.sh'
```
It pulls whatever Colab reached and resumes. Check who holds the lock anytime:
```bash
wsl -d Ubuntu-24.04 -- bash -lc 'cd /home/byapak/voicemodel && \
  HF_TOKEN=$(cat .hf_token) HF_REPO=$(cat .hf_repo) \
  python /mnt/c/Users/user/Documents/VoiceModel/scripts/hf_sync.py status'
```

## Why no progress is lost
The checkpoint carries **weights + optimizer + epoch + step**, so whoever resumes continues
bit-for-bit. The relay refuses to pull an *older* checkpoint over newer local work (it compares
epochs), and Colab pushes every ~20 min, so an abrupt disconnect costs at most a few minutes.
