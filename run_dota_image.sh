#! /bin/bash

export EGL_PLATFORM=surfaceless

DATADIR=~/datasets/dota_v1_val_458
MODELDIR=~/new_model

MODELFILE=$MODELDIR/yolov8n-obb-int8.tflite
BATCHNUM=16

echo $MODELFILE

rm -f ./detection_result_dota.json

UNAME_M=$(uname -m)

if [ "$UNAME_M" = "aarch64" ]; then
    taskset -c 4,5,6,7 ./pkshin_detect image $MODELFILE dsp $DATADIR/labels.txt $DATADIR/images detection_result_dota.json $BATCHNUM

    echo "Calculating mAP.."
    taskset -c 4,5,6,7 ./cal_mAP_dota.sh detection_result_dota.json
elif [ "$UNAME_M" = "x86_64" ]; then
    ./pkshin_detect image $MODELFILE gpu $DATADIR/labels.txt $DATADIR/images detection_result_dota.json $BATCHNUM

    echo "Calculating mAP.."
    ./cal_mAP_dota.sh detection_result_dota.json
else
    echo "    - Error: Unsupported architecture '$UNAME_M'" >&2
    exit 1
fi
