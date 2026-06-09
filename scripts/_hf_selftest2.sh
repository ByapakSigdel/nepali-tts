#!/usr/bin/env bash
# Validate the hardened sync engine: CAS lock, heartbeat, fail-closed reads, pull error paths,
# and the push-ckpt overwrite guard. Uses a tiny fake ckpt so NO 882MB upload happens.
set -uo pipefail
VM="$HOME/voicemodel"
source "$VM/.venv/bin/activate"
export HF_TOKEN="$(cat "$VM/.hf_token")"
export HF_REPO="$(cat "$VM/.hf_repo")"
export DEVICE_ID=selftest
export LOCK_STALE=900
HS="python /mnt/c/Users/user/Documents/VoiceModel/scripts/hf_sync.py"

echo "== status =="; $HS status
echo "== remote-epoch (expect a real integer ~235) =="; $HS remote-epoch
echo "== claim (CAS) =="; $HS claim; echo "claim rc=$?"
echo "== status (expect held by selftest) =="; $HS status
echo "== heartbeat (expect rc 0) =="; $HS heartbeat; echo "hb rc=$?"
echo "== pull config.json (small success path) =="; $HS pull config.json /tmp/cfg_test.json; echo "pull rc=$?"
echo "== pull nonexistent (expect ABSENT, rc 0) =="; $HS pull does-not-exist.bin /tmp/nope.bin; echo "pull rc=$?"
echo "== push-ckpt GUARD: fake epoch-10 ckpt vs remote ~235 -> expect REFUSE rc 9 =="
printf 'notarealcheckpoint' > /tmp/fake.ckpt
$HS push-ckpt /tmp/fake.ckpt 10; echo "push-ckpt rc=$? (expect 9 = refused)"
echo "== release =="; $HS release; echo "rel rc=$?"
echo "== status (expect FREE) =="; $HS status
rm -f /tmp/fake.ckpt /tmp/cfg_test.json /tmp/nope.bin
echo "ALL DONE"
