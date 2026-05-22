#! /bin/sh

DATADIR=~/datasets/coco_val2017
MODELDIR=~/new_model

MODELFILE=$MODELDIR/yolov8x.mxq
VIDEOFILE=~/test_video.mp4

BATCHNUM=16

UNAME_M=$(uname -m)

if [ "$UNAME_M" = "aarch64" ]; then
    taskset -c 4,5,6,7 ./pkshin_detect demo $MODELFILE dsp $DATADIR/labels.txt $VIDEOFILE $BATCHNUM
elif [ "$UNAME_M" = "x86_64" ]; then
    ./pkshin_detect demo $MODELFILE gpu $DATADIR/labels.txt $VIDEOFILE $BATCHNUM
else
    echo "    - Error: Unsupported architecture '$UNAME_M'" >&2
    exit 1
fi
