#! /bin/bash

export EGL_PLATFORM=surfaceless

DATADIR=~/datasets/coco_val2017
MODELDIR=~/new_model

BATCHNUM=6

for MODELFILE in yolov8n.mxq yolov8s.mxq yolov8m.mxq yolov8l.mxq yolov8x.mxq yolov9c.mxq yolov10n.mxq yolov10s.mxq yolov10b.mxq yolov10x.mxq yolov11n.mxq yolov11s.mxq yolov11m.mxq yolov11l.mxq yolov11x.mxq \
yolov8n.tflite yolov8s.tflite yolov8m.tflite yolov8l.tflite yolov8x.tflite yolov9c.tflite yolov10n.tflite yolov10s.tflite yolov10b.tflite yolov10x.tflite yolov11n.tflite yolov11s.tflite yolov11m.tflite yolov11l.tflite yolov11x.tflite
do

MODELFILE=$MODELDIR/$MODELFILE

rm -f detection_result.json

echo $MODELFILE
echo $MODELFILE >> result.txt

UNAME_M=$(uname -m)

if [ "$UNAME_M" = "aarch64" ]; then
    taskset -c 4,5,6,7 ./pkshin_detect image $MODELFILE dsp $DATADIR/labels.txt $DATADIR/300images detection_result.json $BATCHNUM

    echo "Calculating mAP.."
    taskset -c 4,5,6,7 ./cal_mAP_coco.sh detection_result.json
elif [ "$UNAME_M" = "x86_64" ]; then
    ./pkshin_detect image $MODELFILE gpu $DATADIR/labels.txt $DATADIR/300images detection_result.json $BATCHNUM
    
    #echo "Calculating mAP.."
    #./cal_mAP_coco.sh detection_result.json
else
    echo "    - Error: Unsupported architecture '$UNAME_M'" >&2
    exit 1
fi

done
