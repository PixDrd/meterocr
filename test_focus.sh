#!/bin/bash
# Test focus values for a V4L2 webcam.
# Sets focus_absolute in steps of 5 across the given range and captures a
# frame at each value.
#
# Usage: ./test_focus.sh <video_device> [output_dir] [focus_start] [focus_end]
# Example: ./test_focus.sh /dev/v4l/by-id/usb-046d_B525_HD_Webcam_7FDE8130-video-index0 focus_test/ 0 100

set -euo pipefail

STEP=5
SLEEP_SECONDS=2
CAPTURE_RETRIES=3

device="${1:?Usage: $0 <video_device> [output_dir] [focus_start] [focus_end]}"
output_dir="${2:-.}"

mkdir -p "$output_dir"

# Disable continuous autofocus so manual values take effect
echo "Disabling autofocus on $device"
v4l2-ctl -d "$device" --set-ctrl=focus_automatic_continuous=0

# Query focus_absolute range (line looks like: "min=0 max=255 step=1 ...")
ctrl_line=$(v4l2-ctl -d "$device" --list-ctrls | grep focus_absolute)
cam_min=$(echo "$ctrl_line" | sed 's/.*min=\([0-9]*\).*/\1/')
cam_max=$(echo "$ctrl_line" | sed 's/.*max=\([0-9]*\).*/\1/')

focus_start="${3:-$cam_min}"
focus_end="${4:-$cam_max}"

echo "focus_absolute range: $cam_min .. $cam_max (camera)"
echo "Sweeping:             $focus_start .. $focus_end (step $STEP)"
echo "Output:               $output_dir/"
echo ""

focus=$focus_start
while [ "$focus" -le "$focus_end" ]; do
    echo -n "focus_absolute=$focus -> "
    v4l2-ctl -d "$device" --set-ctrl=focus_absolute="$focus"
    sleep "$SLEEP_SECONDS"

    output="${output_dir}/f${focus}.jpg"
    attempt=1
    while [ "$attempt" -le "$CAPTURE_RETRIES" ]; do
        if python capture_frame_stable.py "$device" "$output"; then
            break
        fi
        echo "  capture failed (attempt $attempt/$CAPTURE_RETRIES), retrying..."
        attempt=$((attempt + 1))
        sleep 1
    done
    if [ "$attempt" -gt "$CAPTURE_RETRIES" ]; then
        echo "  ERROR: capture failed after $CAPTURE_RETRIES attempts, skipping focus=$focus"
    fi

    focus=$((focus + STEP))
done

echo ""
echo "Done. Images saved to $output_dir/"
