#!/usr/bin/env python3
"""
hf_sync.py — the "baton" for checkpoint-relay training across machines.

The home laptop and a Google Colab session train ONE shared run by passing the full training
checkpoint (last.ckpt, ~882MB, weights+optimizer+epoch+step) through a private Hugging Face repo.
A LOCK.json enforces "take turns"; PROGRESS.json records the epoch the cloud checkpoint reached.

The USER'S #1 HARD REQUIREMENT IS: NO TRAINING PROGRESS MAY EVER BE LOST. This module is written
to FAIL CLOSED toward that goal. The crucial safety net is the push-side guard in `push-ckpt`:
a machine REFUSES to upload a checkpoint whose epoch is lower than the one already in the cloud.
So even if the lock ever fails and both machines train at once, neither can overwrite the other's
NEWER checkpoint — the worst case degrades from "lost progress" to merely "wasted compute".

  Commands
    push-ckpt <local_ckpt> [epoch|auto] [--config <path>]
                               ATOMIC: upload last.ckpt + PROGRESS.json (+config.json) in ONE commit,
                               with the overwrite guard. Epoch is read FROM the checkpoint bytes
                               ('auto') so the marker can never disagree with the uploaded weights.
    pull <repo_path> <local>   download the latest version of a file (fails non-zero on a real error,
                               returns 0 + prints ABSENT only when the file genuinely doesn't exist)
    push <local> <repo_path>   generic single-file upload (used for config.json seeding, etc.)
    claim                      take the lock  (exit 0 = got it, 3 = busy/uncertain -> FAIL CLOSED)
    heartbeat                  refresh the lock ts (exit 0 ok, 1 = we don't hold it, 2 = transient)
    release                    give up the lock (only if we still hold it)
    status                     print who holds the lock
    remote-epoch               print the cloud checkpoint's epoch (-1 if none), exit 5 on real error
    push-progress <epoch>      (legacy) write PROGRESS.json only — prefer push-ckpt

  Environment
    HF_TOKEN    (required)  a *write* access token (hf_...)
    HF_REPO     (required)  e.g. "yourname/nepali-tts-ckpt"  (created automatically)
    DEVICE_ID   (optional)  label for this machine in the lock; default = hostname
    LOCK_STALE  (optional)  seconds before a held lock is considered abandoned; default 900
    FORCE       (optional)  "1" lets push-ckpt overwrite a NEWER cloud ckpt (dangerous; manual only)
"""
import io
import json
import os
import shutil
import sys
import time
import uuid

REPO_TYPE = "model"
LOCK_PATH = "LOCK.json"
PROGRESS_PATH = "PROGRESS.json"
CKPT_PATH = "last.ckpt"
NONCE_FILE = os.path.expanduser("~/.nepali_tts_lock_nonce")  # shared across claim/heartbeat/release


class LockUnknown(Exception):
    """We could not determine the remote state (transient error). Callers must FAIL CLOSED."""


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
    return int(os.environ.get("LOCK_STALE", "900"))


def _forced():
    return os.environ.get("FORCE", "0") == "1"


def _ensure_repo(api):
    api.create_repo(_repo(), repo_type=REPO_TYPE, private=True, exist_ok=True)


def _is_absent(exc):
    # error classes moved package between huggingface_hub versions (utils -> errors in 1.x)
    try:
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    except ImportError:
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    return isinstance(exc, (EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError))


def _retry(fn, tries=4, base=1.5):
    """Run fn() with backoff. Re-raise 'absent' immediately; raise LockUnknown after exhausting tries."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if _is_absent(exc):
                raise
            last = exc
            if i < tries - 1:
                time.sleep(base ** i)
    raise LockUnknown(str(last))


def _head_sha(api):
    try:
        return api.repo_info(_repo(), repo_type=REPO_TYPE).sha
    except Exception:  # noqa: BLE001
        return None


def _download_json(api, repo_path, revision=None):
    """Return parsed JSON dict, or None if the file is genuinely absent. Raise LockUnknown on error."""
    from huggingface_hub import hf_hub_download

    def go():
        p = hf_hub_download(
            _repo(), repo_path, repo_type=REPO_TYPE, token=_need("HF_TOKEN"),
            force_download=True, revision=revision,
        )
        with open(p) as f:
            return json.load(f)
    try:
        return _retry(go)
    except Exception as exc:  # noqa: BLE001
        if _is_absent(exc):
            return None
        raise LockUnknown(str(exc))


def _commit(api, ops, message, parent=None):
    """Atomic multi-file commit with CAS via parent_commit. Returns True on success.
    Raises LockUnknown on a genuine conflict (parent moved) or transient error."""
    def go():
        return api.create_commit(
            _repo(), operations=ops, commit_message=message,
            repo_type=REPO_TYPE, parent_commit=parent,
        )
    try:
        _retry(go, tries=3)
        return True
    except Exception as exc:  # noqa: BLE001
        raise LockUnknown(str(exc))


def _add(path_in_repo, data):
    from huggingface_hub import CommitOperationAdd
    return CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=data)


# ----------------------------------------------------------------------------- lock

def _my_nonce(create=False):
    if create:
        n = uuid.uuid4().hex
        with open(NONCE_FILE, "w") as f:
            f.write(n)
        return n
    try:
        with open(NONCE_FILE) as f:
            return f.read().strip()
    except OSError:
        return ""


def _write_lock(api, holder, ts, nonce, parent):
    body = json.dumps({"holder": holder, "ts": ts, "nonce": nonce}).encode()
    return _commit(api, [_add(LOCK_PATH, io.BytesIO(body))],
                   f"lock: {holder or 'released'} @ {ts}", parent=parent)


def cmd_claim():
    api = _api()
    _ensure_repo(api)
    me, now = _device(), int(time.time())
    head = _head_sha(api)
    try:
        lock = _download_json(api, LOCK_PATH, revision=head)
    except LockUnknown:
        print("hf_sync: CANNOT read lock state — refusing to claim (fail closed)", flush=True)
        return 3
    if lock and lock.get("holder") and lock["holder"] != me:
        age = now - int(lock.get("ts", 0))
        if age < _stale():
            print(f"hf_sync: BUSY — '{lock['holder']}' holds the lock ({age}s ago). "
                  f"Wait, or it auto-frees after {_stale()}s of silence.", flush=True)
            return 3
        print(f"hf_sync: '{lock['holder']}' silent {age}s — reclaiming.", flush=True)
    nonce = _my_nonce(create=True)
    try:
        _write_lock(api, me, now, nonce, parent=head)  # CAS: fails if head moved since our read
    except LockUnknown:
        print("hf_sync: lost the race / write conflict — another machine claimed first.", flush=True)
        return 3
    print(f"hf_sync: lock acquired by '{me}'", flush=True)
    return 0


def cmd_heartbeat():
    api = _api()
    me, nonce = _device(), _my_nonce()
    head = _head_sha(api)
    try:
        lock = _download_json(api, LOCK_PATH, revision=head)
    except LockUnknown:
        print("hf_sync: heartbeat — transient read error, keeping the lock (will retry).", flush=True)
        return 2  # do NOT relinquish on a transient error
    if not lock or lock.get("holder") != me or lock.get("nonce") != nonce:
        print("hf_sync: heartbeat skipped — we no longer hold the lock", flush=True)
        return 1
    try:
        _write_lock(api, me, int(time.time()), nonce, parent=head)
    except LockUnknown:
        print("hf_sync: heartbeat write conflict — will retry next cycle.", flush=True)
        return 2
    return 0


def cmd_release():
    api = _api()
    me, nonce = _device(), _my_nonce()
    head = _head_sha(api)
    try:
        lock = _download_json(api, LOCK_PATH, revision=head)
    except LockUnknown:
        print("hf_sync: cannot read lock to release — leaving it (it will auto-expire).", flush=True)
        return 1
    if lock and (lock.get("holder") != me or lock.get("nonce") != nonce):
        print(f"hf_sync: not releasing — '{lock.get('holder')}' holds it, not us", flush=True)
        return 1
    try:
        _write_lock(api, None, int(time.time()), "", parent=head)
    except LockUnknown:
        print("hf_sync: release conflict — lock will auto-expire instead.", flush=True)
        return 1
    print("hf_sync: lock released", flush=True)
    return 0


def cmd_status():
    api = _api()
    try:
        lock = _download_json(api, LOCK_PATH)
    except LockUnknown:
        print("hf_sync: lock state UNKNOWN (read error)", flush=True)
        return 0
    if not lock or not lock.get("holder"):
        print("hf_sync: lock is FREE", flush=True)
        return 0
    age = int(time.time()) - int(lock.get("ts", 0))
    tag = " (STALE — reclaimable)" if age >= _stale() else ""
    print(f"hf_sync: held by '{lock['holder']}', last heartbeat {age}s ago{tag}", flush=True)
    return 0


# ----------------------------------------------------------------------------- epoch / progress

def _ckpt_epoch(path):
    """Read the epoch embedded in a Lightning checkpoint, or None if torch can't load it."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    for kw in ({"mmap": True, "weights_only": False}, {"weights_only": False}, {}):
        try:
            ck = torch.load(path, map_location="cpu", **kw)
            ep = ck.get("epoch")
            return int(ep) if ep is not None else None
        except Exception:  # noqa: BLE001
            continue
    return None


def _remote_epoch(api):
    """Cloud checkpoint epoch: -1 if no marker. Raises LockUnknown on a real read error."""
    prog = _download_json(api, PROGRESS_PATH)
    if prog is None:
        return -1
    return int(prog.get("epoch", -1))


def cmd_remote_epoch():
    api = _api()
    try:
        print(_remote_epoch(api))
        return 0
    except LockUnknown:
        print("ERR", flush=True)
        return 5  # caller MUST treat non-integer / exit!=0 as "unknown", never as -1


def cmd_push_progress(epoch):
    """Legacy: write PROGRESS.json only. Prefer push-ckpt (atomic)."""
    api = _api()
    _ensure_repo(api)
    body = json.dumps({"epoch": int(epoch), "device": _device(), "ts": int(time.time())}).encode()
    _commit(api, [_add(PROGRESS_PATH, io.BytesIO(body))], f"{_device()}: progress -> epoch {epoch}",
            parent=_head_sha(api))
    print(f"hf_sync: remote progress marker set to epoch {epoch}", flush=True)
    return 0


def cmd_push_ckpt(local_ckpt, epoch_arg="auto", config_path=None):
    """ATOMIC, GUARDED checkpoint push: upload last.ckpt + PROGRESS.json (+config.json) in one commit.
    Refuses to overwrite a NEWER cloud checkpoint (the core no-progress-loss guard)."""
    api = _api()
    _ensure_repo(api)
    if not os.path.exists(local_ckpt):
        print(f"hf_sync: push-ckpt — no local checkpoint at {local_ckpt}", flush=True)
        return 7

    # 1) Snapshot to an immutable copy so the epoch we read and the bytes we upload are the SAME file.
    stage = local_ckpt + ".staged"
    shutil.copyfile(local_ckpt, stage)
    try:
        # 2) Derive the epoch from the checkpoint bytes (never from the differently-cadenced status.txt).
        epoch = _ckpt_epoch(stage)
        if epoch is None:
            try:
                epoch = int(epoch_arg)
            except (TypeError, ValueError):
                epoch = -1
        if epoch < 0:
            print(f"hf_sync: push-ckpt — refusing, nonsensical epoch {epoch}", flush=True)
            return 7

        # 3) GUARD: never overwrite a strictly-newer cloud checkpoint (unless explicitly FORCEd).
        try:
            remote_ep = _remote_epoch(api)
        except LockUnknown:
            if not _forced():
                print("hf_sync: push-ckpt — cannot read remote epoch; skipping push (will retry).",
                      flush=True)
                return 8
            remote_ep = -1
        if not _forced() and remote_ep >= 0 and epoch < remote_ep:
            print(f"hf_sync: push-ckpt — REFUSING: local epoch {epoch} < remote {remote_ep} "
                  f"(would lose cloud progress). Set FORCE=1 only if you are certain.", flush=True)
            return 9

        # 4) One atomic commit: checkpoint + progress marker (+ config) together.
        prog = json.dumps({"epoch": epoch, "device": _device(), "ts": int(time.time())}).encode()
        ops = [_add(CKPT_PATH, stage), _add(PROGRESS_PATH, io.BytesIO(prog))]
        if config_path and os.path.exists(config_path):
            ops.append(_add("config.json", config_path))
        mb = os.path.getsize(stage) / 1e6
        print(f"hf_sync: push-ckpt epoch {epoch} ({mb:.0f} MB){' +config' if len(ops) > 2 else ''} "
              f"-> {_repo()} ...", flush=True)
        try:
            _commit(api, ops, f"{_device()}: checkpoint @ epoch {epoch}", parent=_head_sha(api))
        except LockUnknown as exc:
            print(f"hf_sync: push-ckpt FAILED to commit ({exc})", flush=True)
            return 6
        print(f"hf_sync: push-ckpt done @ epoch {epoch}", flush=True)
        return 0
    finally:
        try:
            os.remove(stage)
        except OSError:
            pass


# ----------------------------------------------------------------------------- generic file io

def cmd_push(local, repo_path):
    api = _api()
    _ensure_repo(api)
    if not os.path.exists(local):
        sys.exit(f"hf_sync push: local file not found: {local}")
    mb = os.path.getsize(local) / 1e6
    print(f"hf_sync: pushing {local} ({mb:.0f} MB) -> {_repo()}/{repo_path} ...", flush=True)
    try:
        _commit(api, [_add(repo_path, local)], f"{_device()}: update {repo_path}", parent=_head_sha(api))
    except LockUnknown as exc:
        print(f"hf_sync: push FAILED ({exc})", flush=True)
        return 6
    print("hf_sync: push done", flush=True)
    return 0


def cmd_pull(repo_path, local):
    """Download repo_path -> local. Returns 0 + prints ABSENT if it genuinely doesn't exist,
    but exits non-zero on a real download error (so callers never mistake a failed pull for 'fresh')."""
    from huggingface_hub import hf_hub_download
    _ensure_repo(_api())
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)

    def go():
        return hf_hub_download(_repo(), repo_path, repo_type=REPO_TYPE,
                               token=_need("HF_TOKEN"), force_download=True)
    try:
        p = _retry(go)
    except Exception as exc:  # noqa: BLE001
        if _is_absent(exc):
            print(f"hf_sync: ABSENT — no remote {repo_path} (fresh start)", flush=True)
            return 0
        print(f"hf_sync: pull FAILED for {repo_path} ({exc})", flush=True)
        return 6
    shutil.copyfile(p, local)
    if os.path.getsize(local) <= 0:
        print(f"hf_sync: pull produced an empty file for {repo_path}", flush=True)
        return 6
    mb = os.path.getsize(local) / 1e6
    print(f"hf_sync: pulled {repo_path} ({mb:.0f} MB) -> {local}", flush=True)
    return 0


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, rest = sys.argv[1], sys.argv[2:]

    def push_ckpt():
        local = rest[0]
        epoch_arg = rest[1] if len(rest) > 1 and rest[1] != "--config" else "auto"
        config_path = None
        if "--config" in rest:
            config_path = rest[rest.index("--config") + 1]
        return cmd_push_ckpt(local, epoch_arg, config_path)

    table = {
        "push-ckpt": push_ckpt,
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
