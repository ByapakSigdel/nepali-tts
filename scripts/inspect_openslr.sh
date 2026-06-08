#!/usr/bin/env bash
# Inspect the Nepali OpenSLR datasets: list every download file and its real size (via HTTP HEAD).
set -uo pipefail

grand=0
for n in 43 54 143; do
  echo "================ SLR${n} : https://www.openslr.org/${n}/ ================"
  html=$(curl -sL "https://www.openslr.org/${n}/")
  echo "$html" | grep -oiE '<title>[^<]*</title>' | head -1 | sed -E 's@</?title>@@gI'

  links=$(echo "$html" | grep -oiE 'href="[^"]*"' \
            | sed -E 's/href="//I; s/"$//' \
            | grep -iE "/resources/${n}/" | sort -u)

  if [ -z "$links" ]; then echo "  (no resource links found)"; echo; continue; fi

  sub=0
  while IFS= read -r url; do
    [ -z "$url" ] && continue
    case "$url" in
      http*) full="$url" ;;
      /*)    full="https://www.openslr.org${url}" ;;
      *)     full="https://www.openslr.org/${url}" ;;
    esac
    len=$(curl -sIL "$full" | grep -i '^content-length:' | tail -1 | tr -dc '0-9')
    [ -z "$len" ] && len=0
    mb=$(( len / 1048576 ))
    sub=$(( sub + len ))
    printf "  %7d MB   %s\n" "$mb" "$full"
  done <<< "$links"

  printf ">> SLR%s subtotal: %d MB (~%d GB)\n\n" "$n" "$(( sub/1048576 ))" "$(( sub/1073741824 ))"
  grand=$(( grand + sub ))
done
printf "================ GRAND TOTAL: %d MB (~%d GB) ================\n" "$(( grand/1048576 ))" "$(( grand/1073741824 ))"
