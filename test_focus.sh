#!/bin/bash
# Test focus values for a V4L2 webcam.
# Sets focus_absolute in steps of 5 across the full supported range and
# captures a frame at each value.
#
# Usage: ./test_focus.sh <video_device>
# Example: ./test_focus.sh /dev/v4l/by-id/usb-046d_B525_HD_Webcam_7FDE8130-video-index0

set -euo pipefail

STEP=5
SLEEP_SECONDS=2

device="${1:?Usage: $0 <video_device>}"

# Disable autofocus so manual values take effect
echo "Disabling autofocus on $device"
v4l2-ctl -d "$device" --set-ctrl=focus_auto=0

# Query focus_absolute range (line looks like: "min=0 max=255 step=1 ...")
ctrl_line=$(v4l2-ctl -d "$device" --list-ctrls | grep focus_absolute)
focus_min=$(echo "$ctrl_line" | sed 's/.*min=\([0-9]*\).*/\1/')
focus_max=$(echo "$ctrl_line" | sed 's/.*max=\([0-9]*\).*/\1/')

echo "focus_absolute range: $focus_min .. $focus_max (stepping by $STEP)"
echo ""

focus=$focus_min
while [ "$focus" -le "$focus_max" ]; do
    echo -n "focus_absolute=$focus -> "
    v4l2-ctl -d "$device" --set-ctrl=focus_absolute="$focus"
    sleep "$SLEEP_SECONDS"
    python capture_frame_stable.py "$device" "f${focus}.png"
    focus=$((focus + STEP))
done

echo ""
echo "Done. Images saved as f<value>.png"
