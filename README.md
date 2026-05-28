# Multicam-perception

A high-performance, multi-threaded object detection and 3D perception engine built using LiteRT(TFLite) API. Designed for edge devices and embedded platforms, this application supports a wide range of state-of-the-art models, hardware accelerators (GPU, DSP), zero-copy Wayland/EGL rendering, and multi-camera 3D fusion using the nuScenes dataset.

## 🚀 Key Features

*   **Broad Model Support:** Compatible with SSD MobileNet, EfficientDet, YOLO (v3 ~ v12, including OBB), and DETR (ResNet, RT-DETR).
*   **Hardware Acceleration:** Supports CPU, GPU (with GL fallback for ARM64), and Hexagon DSP delegates.
*   **Multi-Threaded Pipeline:** Implements a custom ThreadPool for asynchronous pre-processing, inference, and post-processing to maximize throughput.
*   **Zero-Copy Display rendering:** Direct rendering to Wayland surfaces using OpenGL ES 2.0 and EGL.
*   **3D Multi-Camera Fusion:** 
    *   Full support for parsing and evaluating the nuScenes dataset.
    *   Implementation of 3D Weighted Box Fusion (WBF).
    *   Built-in Multi-Object Tracker (MOT) with Kalman Filtering and Mahalanobis distance matching.
*   **GStreamer Integration:** Hardware-accelerated video decoding for continuous inference.
*   **QCarCam Support:** Direct integration with Qualcomm camera APIs for real-time edge processing.

## 🛠️ Dependencies

Ensure the following libraries are installed on your target system:
*   TFLite (LiteRT) (`libtensorflowlite.so`, `libtensorflowlite_gpu_delegate.so`, `libtensorflowlite_hexagon_delegate.so`)
*   TFLite-for-Heterogeneous-Accelerators (Need for supporting Hailo and Maccel NPUs)
*   OpenCV (`libopencv_core`, `libopencv_imgproc`, `libopencv_dnn`, etc.)
*   GStreamer 1.0 (`gstreamer-1.0`, `gstreamer-app-1.0`, `gstreamer-video-1.0`)
*   Wayland & EGL (`wayland-client`, `wayland-egl`, `EGL`, `GLESv2`)
*   libturbojpeg (`libturbojpeg.so`)
*   JSON-C (`libjson-c.so`)
*   Eigen3 (Header only)
*   *Platform Specific:* `qcarcam` and `fastcv` (for ARM64/Qualcomm environments).

## 💻 Usage

The executable `pkshin_detect` provides five distinct operational modes.

### 1. Camera Mode (QCarCam API)
Runs object detection on real-time camera feeds.  

./pkshin_detect camera [MODEL] [ACCELERATOR] [LABEL] [DISPLAY]

### 2. Image Mode
Runs object detection on a directory of JPEG images and outputs a JSON result.  

./pkshin_detect image [MODEL] [ACCELERATOR] [LABEL] [IMG_DIR] [RESULT] [BATCH_SIZE] [SCORE_THRESHOLDS]

### 3. Video Mode
Runs object detection on a video file using GStreamer for decoding and Wayland for display.  

./pkshin_detect video [MODEL] [ACCELERATOR] [LABEL] [VIDEO_PATH] [BATCH_SIZE]

### 4. Demo Mode
Runs a side-by-side comparison of two models/interpreters on a video file.  

./pkshin_detect demo [MODEL] [ACCELERATOR] [LABEL] [VIDEO_PATH] [BATCH_SIZE]

### 5. Multicam Mode (nuScenes 3D Detection)
Runs 3D object detection and tracking using synchronized multi-camera datasets (e.g., nuScenes). Evaluates metrics (mAP, NDS, etc.).  

./pkshin_detect multicam [MODEL] [ACCELERATOR] [DATASET_DIR] [FUSION_MODE] [RESULT_PATH] [EXEC_TIMES]  
[FUSION_MODE]: Use sequential or simultaneous for Late Fusion strategy.  
[EXEC_TIMES]: (Optional) Comma-separated list of 4 execution time parameters.

## 🧠 Supported Models
The engine automatically parses the model type from the filename. Ensure your .tflite file contains the appropriate keyword:  
* ssd_mobilenet  
* efficientdet, efficientdet_lite  
* yolov3, yolov5, yolov8, yolov8_obb, yolov9, yolov10, yolov11, yolov11_obb, yolov12  
* detr_resnet, rt_detr

## 📊 3D Tracking & Fusion Architecture
The multicam mode includes a sophisticated perception stack:

* 2D to 3D Projection: Projects 2D bounding boxes into 3D global space using camera intrinsics, ego-pose transformations, and nuScenes class priors.  
* Weighted Box Fusion (WBF): Merges overlapping overlapping predictions across multiple camera views dynamically.  
* Kalman Filter Tracking: Predicts object motion (velocity, yaw) and smooths trajectories over time to handle occlusions.  
* Devkit Evaluation: Automatically calculates nuScenes metrics (NDS, mAP, mATE, mASE, mAOE, mAVE, mAAE).
