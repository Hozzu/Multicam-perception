#ifndef _PREPROCESS_HPP_
#define _PREPROCESS_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <engine_interface.hpp>

static bool preprocess(tflite::Interpreter * interpreter, int model_mode, uint8_t * rgb_buf_ptr, int img_height, int img_width, int cur_batch = 0){
    int tflite_input_size = interpreter->tflite_inputs().size();
    int hailo_input_size = interpreter->hailo_inputs().size();
    int maccel_input_size = interpreter->maccel_inputs().size();

    if(tflite_input_size > 1 || hailo_input_size > 1 || maccel_input_size > 1){
        std::cerr << "ERROR: Preprocess only supports 1 input\n";
        return false;
    }
    
    // Determine mean and stddev based on model mode outside the loop
    float mean_rgb[3] = {0.0f, 0.0f, 0.0f};
    float stddev_rgb[3] = {1.0f, 1.0f, 1.0f};

    switch(model_mode){
        case 1: // ssd_mobilenet
            mean_rgb[0] = 127.5f; mean_rgb[1] = 127.5f; mean_rgb[2] = 127.5f;
            stddev_rgb[0] = 127.5f; stddev_rgb[1] = 127.5f; stddev_rgb[2] = 127.5f;
            break;
        case 2: // efficientdet
        case 7: // detr resnet
            mean_rgb[0] = 0.485f * 255.0f; mean_rgb[1] = 0.456f * 255.0f; mean_rgb[2] = 0.406f * 255.0f;
            stddev_rgb[0] = 0.229f * 255.0f; stddev_rgb[1] = 0.224f * 255.0f; stddev_rgb[2] = 0.225f * 255.0f;
            break;
        case 3: // efficientdet lite
            mean_rgb[0] = 127.0f; mean_rgb[1] = 127.0f; mean_rgb[2] = 127.0f;
            stddev_rgb[0] = 128.0f; stddev_rgb[1] = 128.0f; stddev_rgb[2] = 128.0f;
            break;
        case 4: // yolo
        case 5: // yolov10
        case 6: // yolo obb
        case 8: // rt detr
            mean_rgb[0] = 0.0f; mean_rgb[1] = 0.0f; mean_rgb[2] = 0.0f;
            stddev_rgb[0] = 255.0f; stddev_rgb[1] = 255.0f; stddev_rgb[2] = 255.0f;
            break;
        default:
            std::cerr << "ERROR: Unsupported model mode in preprocess " << model_mode << "\n";
            return false;
    }

    cv::Mat cvImg(cv::Size(img_width, img_height), CV_8UC3, rgb_buf_ptr);
    cv::Mat resizecvImg;
    cv::Mat floatImg;
    
    // Caching variables to avoid redundant computation if multiple inputs have the same dimensions
    int cached_input_height = -1;
    int cached_input_width = -1;
    bool is_float_prepared = false;

    for(int i = 0; i < interpreter->inputs().size(); i++){
        TfLiteTensor* cur_tensor = interpreter->input_tensor(i);
        TfLiteIntArray* input_dims = cur_tensor->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];
        int input_channel = input_dims->data[3];

        if (i > 0 && input_height == cached_input_height && input_width == cached_input_width && cur_tensor->type == interpreter->input_tensor(0)->type) {
            size_t num_elements = input_height * input_width * input_channel;
            if (cur_tensor->type == kTfLiteFloat32) {
                float* src_ptr = interpreter->typed_input_tensor<float>(0, cur_batch);
                float* dst_ptr = interpreter->typed_input_tensor<float>(i, cur_batch);
                memcpy(dst_ptr, src_ptr, num_elements * sizeof(float));
            } else if (cur_tensor->type == kTfLiteUInt8) {
                uint8_t* src_ptr = interpreter->typed_input_tensor<uint8_t>(0, cur_batch);
                uint8_t* dst_ptr = interpreter->typed_input_tensor<uint8_t>(i, cur_batch);
                memcpy(dst_ptr, src_ptr, num_elements * sizeof(uint8_t));
            }
            continue; // Skip the rest of the scaling and conversion process for this input
        }

        // Calculate scale
        float scale_height = (float) input_height / img_height;
        float scale_width = (float) input_width / img_width;

        switch(model_mode){
            case 1: // ssd_mobilenet
            case 7: // detr_resnet
            case 8: // rt detr
                break;
            default:
            {
                float scale = scale_height < scale_width ? scale_height : scale_width;
                scale_height = scale;
                scale_width = scale;
            }
        }

        // Resize image
        int resize_height = img_height * scale_height;
        int resize_width = img_width * scale_width;

        // Only perform cv::resize if the target dimensions changed (or if it's the first time)
        if (input_height != cached_input_height || input_width != cached_input_width) {
            cv::resize(cvImg, resizecvImg, cv::Size(resize_width, resize_height));
            is_float_prepared = false; // Reset float cache
            cached_input_height = input_height;
            cached_input_width = input_width;
        }
        
        // Normalize and pad input
        if(cur_tensor->type == kTfLiteUInt8){
            uint8_t * input_img_ptr = interpreter->typed_input_tensor<uint8_t>(i, cur_batch);
            memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

            for(int j = 0; j < resize_height; j++){
                memcpy(&input_img_ptr[j * input_width * 3], resizecvImg.ptr<uint8_t>(j), resize_width * 3 * sizeof(uint8_t));
            }
        }
        else if(cur_tensor->type == kTfLiteFloat32){
            if (!is_float_prepared) {
                resizecvImg.convertTo(floatImg, CV_32FC3);
                cv::subtract(floatImg, cv::Scalar(mean_rgb[0], mean_rgb[1], mean_rgb[2]), floatImg);
                cv::divide(floatImg, cv::Scalar(stddev_rgb[0], stddev_rgb[1], stddev_rgb[2]), floatImg);
                is_float_prepared = true;
            }

            float * input_img_ptr = interpreter->typed_input_tensor<float>(i, cur_batch);
            memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

            for(int j = 0; j < resize_height; j++){
                memcpy(&input_img_ptr[j * input_width * 3], floatImg.ptr<float>(j), resize_width * 3 * sizeof(float));
            }
        }
        else{
            std::cerr << "ERROR: Unsupported input tensor data type in preprocess\n";
            return false;
        }
    }

    return true;
}

#endif  // _PREPROCESS_HPP_