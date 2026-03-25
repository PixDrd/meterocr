#!/bin/bash
SETTINGS="-r 2000x2000 -S 125"
fswebcam $SETTINGS -d /dev/v4l/by-id/usb-046d_B525_HD_Webcam_63928130-video-index0 m1_full.jpg
fswebcam $SETTINGS -d /dev/v4l/by-id/usb-046d_B525_HD_Webcam_7AAA8130-video-index0 m2_full.jpg
fswebcam $SETTINGS -d /dev/v4l/by-id/usb-046d_B525_HD_Webcam_7FDE8130-video-index0 m3_full.jpg
