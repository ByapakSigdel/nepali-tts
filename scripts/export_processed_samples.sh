#!/usr/bin/env bash
# Copy a few processed clips to the Windows eval folder for listening + show manifest sanity.
DST=/mnt/c/Users/user/Documents/VoiceModel/eval/processed
SRC="$HOME/voicemodel/data/processed/wavs"
mkdir -p "$DST"
for s in slr143_female nep_0546 nep_2099 slr143_male; do
  f=$(ls "$SRC" | grep "^${s}__" | head -1)
  if [ -n "$f" ]; then cp "$SRC/$f" "$DST/" && echo "copied $f"; fi
done
echo "--- metadata.csv (first 3 rows) ---"
head -3 "$HOME/voicemodel/data/processed/metadata.csv"
echo "--- processed wav count ---"
ls "$SRC" | wc -l
