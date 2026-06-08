#!/usr/bin/env bash
# Print the OpenSLR "about" summary for each Nepali dataset (strip HTML, keep key fields).
for n in 43 143 54; do
  echo "================ SLR${n} ================"
  curl -sL "https://www.openslr.org/${n}/" \
    | sed -E 's/<[^>]+>/ /g' \
    | tr -s ' ' \
    | grep -iE 'Identifier|Summary|Category|Licen|About this resource|speaker|Nepali|hours|utterance|recordings' \
    | sed -E 's/^ +//' | sort -u | head -20
  echo
done
