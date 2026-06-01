#! /bin/bash

make

export EGL_PLATFORM=surfaceless

DATADIR=~/datasets/nuScenes_mini
MODELDIR=~/new_model

#EXECTIMES=""

set -- $EXECTIMES

for MODELFILE in yolov8n.mxq yolov8s.mxq yolov8m.mxq yolov8l.mxq yolov8x.mxq yolov9c.mxq yolov10n.mxq yolov10s.mxq yolov10b.mxq yolov10x.mxq yolov11n.mxq yolov11s.mxq yolov11m.mxq yolov11l.mxq yolov11x.mxq
#for MODELFILE in yolov8n.tflite yolov8s.tflite yolov8m.tflite yolov8l.tflite yolov8x.tflite yolov9c.tflite yolov10n.tflite yolov10s.tflite yolov10b.tflite yolov10x.tflite yolov11n.tflite yolov11s.tflite yolov11m.tflite yolov11l.tflite yolov11x.tflite
#for MODELFILE in yolov8n.all yolov8s.all yolov8m.all yolov8l.all yolov8x.all yolov9c.all yolov10n.all yolov10s.all yolov10b.all yolov10x.all yolov11n.all yolov11s.all yolov11m.all yolov11l.all yolov11x.all
do

MODELFILE=$MODELDIR/$MODELFILE

rm -f multicam_result.json

echo $MODELFILE
echo $MODELFILE >> result.txt

EXECTIME=$1

shift

echo $EXECTIME

UNAME_M=$(uname -m)

if [ "$UNAME_M" = "aarch64" ]; then
    echo "sequential" >> result.txt
    taskset -c 4,5,6,7 ./pkshin_detect multicam $MODELFILE dsp $DATADIR sequential multicam_result.json $EXECTIME

    echo "simultaneous" >> result.txt
    taskset -c 4,5,6,7 ./pkshin_detect multicam $MODELFILE dsp $DATADIR simultaneous multicam_result.json $EXECTIME
elif [ "$UNAME_M" = "x86_64" ]; then
    echo "sequential" >> result.txt
    ./pkshin_detect multicam $MODELFILE gpu $DATADIR sequential multicam_result.json $EXECTIME

    echo "simultaneous" >> result.txt
    ./pkshin_detect multicam $MODELFILE gpu $DATADIR simultaneous multicam_result.json $EXECTIME
else
    echo "    - Error: Unsupported architecture '$UNAME_M'" >&2
    exit 1
fi

done
