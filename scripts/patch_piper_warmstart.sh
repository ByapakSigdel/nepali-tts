#!/usr/bin/env bash
# SMART warm-start patch for piper1-gpl.
# Stock `_warmstart_vocoder_from_ckpt` copies only the vocoder (model_g.dec/enc_q/flow), discarding the
# text encoder (enc_p) AND the duration predictor (dp) AND the discriminator (model_d) -- i.e. most of
# the learned "how to speak Nepali" knowledge. We broaden KEEP_PREFIXES to ("model_g.", "model_d.") so
# the ENTIRE generator + discriminator transfer. The loader's existing shape check then skips ONLY the
# speaker-embedding table (model_g.emb_g.weight is [n_old, 512] vs [n_new, 512]), so it re-initializes
# fresh for the new speaker set -- exactly what we want.
# Idempotent / re-runnable. Re-apply after any reinstall of piper1-gpl (laptop or Colab).
set -uo pipefail
LIGHT="$HOME/voicemodel/piper1-gpl/src/piper/train/vits/lightning.py"
[ -f "$LIGHT" ] || { echo "!! not found: $LIGHT"; exit 1; }

python3 - "$LIGHT" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
new_block = '''KEEP_PREFIXES = (
            "model_g.",
            "model_d.",
        )'''
s2 = re.sub(r'KEEP_PREFIXES = \([^)]*\)', new_block, s, count=1)
if "KEEP_PREFIXES" not in s:
    print("!! KEEP_PREFIXES not found -- piper layout changed; inspect manually"); sys.exit(1)
if s2 != s:
    open(p, "w").write(s2)
    print(">>> patched: warm-start now copies the full generator + discriminator (emb_g stays fresh)")
else:
    print(">>> already in the desired form")
PY
echo ">>> warm-start KEEP_PREFIXES now:"
sed -n '/KEEP_PREFIXES = (/,/)/p' "$LIGHT"
