#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <folder>"
    exit 1
fi

folder="$1"

for image in "$folder"/*.png; do
    [[ -e "$image" ]] || { echo "No PNG files found in $folder"; exit 1; }
    filename=$(basename "$image" .png)
    meter="${filename%%_*}"
    reading="${filename#*_}"
    reading="${reading%%_*}"
    echo "Labeling $image  meter=$meter  reading=$reading"
    meterocr label-frame --meter "$meter" --image "$image" --reading "$reading"
done
