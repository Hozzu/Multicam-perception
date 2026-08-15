#! /bin/bash

make -j $(nproc)

DATADIR=~/datasets/nuScenes_mini
MODELDIR=~/new_model

for MODELFILE in yolov11x.all
do

MODELFILE=$MODELDIR/$MODELFILE
#EXECTIME="264,1000000,47,1000000"

echo $MODELFILE
echo $EXECTIME

UNAME_M=$(uname -m)

if [ "$UNAME_M" = "aarch64" ]; then
    taskset -c 4,5,6,7 ./pkshin_detect multicam_demo $MODELFILE dsp $DATADIR sequential video.mp4 $EXECTIME
    #taskset -c 4,5,6,7 ./pkshin_detect multicam_demo $MODELFILE dsp $DATADIR simultaneous video2.mp4 $EXECTIME
elif [ "$UNAME_M" = "x86_64" ]; then
    ./pkshin_detect multicam_demo $MODELFILE gpu $DATADIR sequential video_our.mp4 $EXECTIME
    #./pkshin_detect multicam_demo $MODELFILE gpu $DATADIR simultaneous video2.mp4 $EXECTIME
else
    echo "    - Error: Unsupported architecture '$UNAME_M'" >&2
    exit 1
fi

done
