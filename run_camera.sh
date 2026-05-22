#! /bin/bash

DATADIR=~/datasets/coco_val2017
MODELDIR=~/model

MODELFILE=$MODELDIR/yolov8n.hef

UNAME_M=$(uname -m)

if [ "$UNAME_M" = "aarch64" ]; then
    taskset -c 4,5,6,7 ./pkshin_detect camera $MODELFILE dsp $DATADIR/labels.txt ssdDisplay.xml
elif [ "$UNAME_M" = "x86_64" ]; then
    ./pkshin_detect camera $MODELFILE gpu $DATADIR/labels.txt ssdDisplay.xml
else
    echo "    - Error: Unsupported architecture '$UNAME_M'" >&2
    exit 1
fi