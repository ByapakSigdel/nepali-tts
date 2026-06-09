#!/usr/bin/env python3
"""
hf_sync.py — the "baton" for checkpoint-relay training across machines.

Both the home laptop and a Google Colab session run the SAME training run by passing
the full training checkpoint (last.ckpt) through a private Hugging Face Hub repo.
A tiny LOCK.json in that repo enforces "take turns" so the two machines can never
train at once and clobber each other's progress.

  Commands
    push <local> <repo_path>   upload a file (e.g. last.ckpt) to the hub repo
    pull <repo_path> <local>   download the latest version of a file from the repo
    claim                      try to take the training lock (exit 0 = got it, 3 = busy)
    heartbeat                  refresh the lock timestamp (call periodically while training)
    release                    give up the lock
    status                     print who holds the lock and how old it is

  Environment
    HF_TOKEN    (required)  a *write* access token (hf_...)
    HF_REPO     (required)  e.g. "yourname/nepali-tts-ckpt"  (created automatically)
    DEVICE_ID   (optional)  label for this machine in the lock; default = hostname
    LOCK_STALE  (optional)  seconds before a held lock is considered abandoned; default 1800

Design notes (documented on purpose — see docs/DEVLOG.md):
  * The checkpoint carries full state (weights + optimizer + epoch + step), so whoever
    pulls it resumes bit-for-bit. No progress is lost as long as the latest ckpt is
    pushed before a machine stops and pulled before the next starts.
  * Colab can die without warning, so we push every ~20 min there, not just at the end.
  * The lock is "good enough": claim -> wait -> re-read to confirm we still hold it
    (guards the rare race), and a stale lock (no heartbeat for LOCK_STALE secs) is
    auto-reclaimable so a crashed machine can't block the other forever.
"""
import json
import os
import sys
import time

REPO_TYPE = "model"
LOCK_PATH = "LOCK.json"


def _need(name):
    v = os.environ.get(name)
    if not v:
        sys.exit(f"hf_sync: environment variable {name} is required")
    return v


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("hf_sync: huggingface_hub not installed — run: pip install huggingface_hub")
    return HfApi(token=_need("HF_TOKEN"))


def _repo():
    return _need("HF_REPO")


def _device():
    import socket
    return os.environ.get("DEVICE_ID") or socket.gethostname()


def _stale():
    return int(os.environ.get("LOCK_STALE", "1800"))


def _ensure_repo(api):
    api.create_repo(_repo(), repo_type=REPO_TYPE, private=True, exist_ok=True)


def _read_lock(api):
    """Return the current lock dict, or None if there's no lock file yet."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    try:
        p = hf_hub_download(
            _repo(), LOCK_PATH, repo_type=REPO_TYPE, token=_need("HF_TOKEN"),
            force_download=True,  # never trust the local cache for the lock
        )
        with open(p) as f:
            return json.load(f)
    except (EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError):
        return None
    except Exception:
        return None


def _write_lock(api, holder, ts):
    import io
    body = json.dumps({"holder": holder, "ts": ts}).encode()
    api.upload_file(
        path_or_fileobj=io.BytesIO(body), path_in_repo=LOCK_PATH,
        repo_id=_repo(), repo_type=REPO_TYPE,
        commit_message=f"lock: {holder if holder else 'released'} @ {ts}",
    )


def cmd_push(local, repo_path):
    api = _api()
    _ensure_repo(api)
    if not os.path.exists(local):
        sys.exit(f"hf_sync push: local file not found: {local}")
    mb = os.path.getsize(local) / 1e6
    print(f"hf_sync: pushing {local} ({mb:.0f} MB) -> {_repo()}/{repo_path} ...", flush=True)
    api.upload_file(
        path_or_fileobj=local, path_in_repo=repo_path,
        repo_id=_repo(), repo_type=REPO_TYPE,
        commit_message=f"{_device()}: update {repo_path}",
    )
    print("hf_sync: push done", flush=True)


def cmd_pull(repo_path, local):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    _ensure_repo(_api())
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    try:
        p = hf_hub_download(
            _repo(), repo_path, repo_type=REPO_TYPE, token=_need("HF_TOKEN"),
            force_download=True,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        print(f"hf_sync: no remote {repo_path} yet — nothing to pull (fresh start)", flush=True)
        return 0
    import shutil
    shutil.copyfile(p, local)
    mb = os.path.getsize(local) / 1e6
    print(f"hf_sync: pulled {repo_path} ({mb:.0f} MB) -> {local}", flush=True)
    return 0


def cmd_claim():
    api = _api()
    _ensure_repo(api)
    me, now = _device(), int(time.time())
    lock = _read_lock(api)
    if lock and lock.get("holder") and lock["holder"] != me:
        age = now - int(lock.get("ts", 0))
        if age < _stale():
            print(f"hf_sync: BUSY — '{lock['holder']}' holds the lock ({age}s ago). "
                  f"Wait, or it auto-frees after {_stale()}s of silence.", flush=True)
            return 3
        print(f"hf_sync: '{lock['holder']}' went silent {age}s ago — reclaiming.", flush=True)
    _write_lock(api, me, now)
    time.sleep(5)  # let the commit settle, then confirm we still hold it (race guard)
    lock = _read_lock(api)
    if lock and lock.get("holder") == me:
        print(f"hf_sync: lock acquired by '{me}'", flush=True)
        return 0
    print(f"hf_sync: lost a race for the lock (now '{lock.get('holder') if lock else None}')", flush=True)
    return 3


def cmd_heartbeat():
    api = _api()
    me = _device()
    lock = _read_lock(api)
    if not lock or lock.get("holder") != me:
        print(f"hf_sync: heartbeat skipped — we don't hold the lock", flush=True)
        return 1
    _write_lock(api, me, int(time.time()))
    return 0


def cmd_release():
    api = _api()
    me = _device()
    lock = _read_lock(api)
    if lock and lock.get("holder") not in (None, me):
        print(f"hf_sync: not releasing — '{lock['holder']}' holds it, not us", flush=True)
        return 1
    _write_lock(api, None, int(time.time()))
    print("hf_sync: lock released", flush=True)
    return 0


PROGRESS_PATH = "PROGRESS.json"


def cmd_push_progress(epoch):
    """Record the epoch reached by the checkpoint currently in the repo."""
    import io
    api = _api()
    body = json.dumps({"epoch": int(epoch), "device": _device(), "ts": int(time.time())}).encode()
    api.upload_file(
        path_or_fileobj=io.BytesIO(body), path_in_repo=PROGRESS_PATH,
        repo_id=_repo(), repo_type=REPO_TYPE,
        commit_message=f"{_device()}: progress -> epoch {epoch}",
    )
    print(f"hf_sync: remote progress marker set to epoch {epoch}", flush=True)


def cmd_remote_epoch():
    """Print the epoch of the checkpoint in the repo, or -1 if there is none."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    try:
        p = hf_hub_download(
            _repo(), PROGRESS_PATH, repo_type=REPO_TYPE, token=_need("HF_TOKEN"),
            force_download=True,
        )
        with open(p) as f:
            print(int(json.load(f).get("epoch", -1)))
    except (EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError):
        print(-1)
    except Exception:
        print(-1)
    return 0


def cmd_status():
    lock = _read_lock(_api())
    if not lock or not lock.get("holder"):
        print("hf_sync: lock is FREE", flush=True)
        return 0
    age = int(time.time()) - int(lock.get("ts", 0))
    tag = " (STALE — reclaimable)" if age >= _stale() else ""
    print(f"hf_sync: held by '{lock['holder']}', last heartbeat {age}s ago{tag}", flush=True)
    return 0


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, rest = sys.argv[1], sys.argv[2:]
    table = {
        "push": lambda: cmd_push(rest[0], rest[1]),
        "pull": lambda: cmd_pull(rest[0], rest[1]),
        "claim": cmd_claim,
        "heartbeat": cmd_heartbeat,
        "release": cmd_release,
        "status": cmd_status,
        "push-progress": lambda: cmd_push_progress(rest[0]),
        "remote-epoch": cmd_remote_epoch,
    }
    if cmd not in table:
        sys.exit(f"hf_sync: unknown command '{cmd}'\n{__doc__}")
    rc = table[cmd]()
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
