#!/bin/bash
source .venv/bin/activate
mkdir data/logs
meterocr maybe_upload_latest_images >> data/logs/maybe_upload_latest_images.log
