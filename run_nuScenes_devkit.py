#! /bin/bash
export EGL_PLATFORM=surfaceless

python3 -m nuscenes.eval.detection.evaluate multicam_result.json --eval_set mini_val --dataroot ~/datasets/nuScenes_mini/ --version v1.0-mini
