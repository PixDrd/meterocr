#!/bin/bash
source .venv/bin/activate

LOGDIR="data/logs"
LOG="${LOGDIR}/maybe_upload_latest_images.log"

if [ ! -d "$LOGDIR" ]
then
	mkdir "$LOGDIR"
fi

echo -n "`date` " >> $LOG
meterocr maybe_upload_latest_images >> $LOG
