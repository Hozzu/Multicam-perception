#! /bin/bash

export WAYLAND_DISPLAY=wayland-99
export XDG_RUNTIME_DIR=/run/user/$(id -u)

weston --backend=headless-backend.so --socket=$WAYLAND_DISPLAY &
WESTON_PID=$!

make

DATADIR=~/datasets/nuScenes_mini
MODELDIR=~/new_model

for MODELFILE in yolov8m.all
do

MODELFILE=$MODELDIR/$MODELFILE
EXECTIME="68,1000000,18,1000000"

echo $MODELFILE
echo $EXECTIME

UNAME_M=$(uname -m)

if [ "$UNAME_M" = "aarch64" ]; then
    taskset -c 4,5,6,7 ./pkshin_detect multicam_demo $MODELFILE dsp $DATADIR sequential video.mp4 $EXECTIME
    taskset -c 4,5,6,7 ./pkshin_detect multicam_demo $MODELFILE dsp $DATADIR simultaneous video.mp4 $EXECTIME
elif [ "$UNAME_M" = "x86_64" ]; then
    ./pkshin_detect multicam_demo $MODELFILE gpu $DATADIR sequential video2.mp4 $EXECTIME
    ./pkshin_detect multicam_demo $MODELFILE gpu $DATADIR simultaneous video2.mp4 $EXECTIME
else
    echo "    - Error: Unsupported architecture '$UNAME_M'" >&2
    exit 1
fi

done
