#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <folder>"
    exit 1
fi

folder="$1"

found=0
while IFS= read -r -d '' image; do
    found=1
    filename=$(basename "$image")
    filename="${filename%.*}"
    meter=$(echo "${filename%%_*}" | tr '[:lower:]' '[:upper:]')
    rest="${filename#*_}"   # strip meter prefix
    reading=$(printf '%05d' "${rest#*_}")  # strip timestamp, pad to 5 digits
    echo "Labeling $image  meter=$meter  reading=$reading"
    meterocr label-frame --meter "$meter" --image "$image" --reading "$reading"
done < <(find "$folder" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | sort -z)

if [[ $found -eq 0 ]]; then
    echo "No image files found in $folder"
    exit 1
fi
