#! /bin/bash

export EGL_PLATFORM=surfaceless

DATADIR=~/datasets/coco_val2017
MODELDIR=~/new_model

BATCHNUM=4

for MODELFILE in yolov11x.hef yolov11m.mxq yolov11m.tflite
do

MODELFILE=$MODELDIR/$MODELFILE

rm -f detection_result.json

echo $MODELFILE

UNAME_M=$(uname -m)

if [ "$UNAME_M" = "aarch64" ]; then
    taskset -c 4,5,6,7 ./pkshin_detect image $MODELFILE dsp $DATADIR/labels.txt $DATADIR/300images detection_result.json $BATCHNUM

    echo "Calculating mAP.."
    taskset -c 4,5,6,7 ./cal_mAP_coco.sh detection_result.json
elif [ "$UNAME_M" = "x86_64" ]; then
    ./pkshin_detect image $MODELFILE gpu $DATADIR/labels.txt $DATADIR/300images detection_result.json $BATCHNUM
    
    echo "Calculating mAP.."
    ./cal_mAP_coco.sh detection_result.json
else
    echo "    - Error: Unsupported architecture '$UNAME_M'" >&2
    exit 1
fi

done
