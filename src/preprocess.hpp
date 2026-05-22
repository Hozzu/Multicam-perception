#ifndef _PREPROCESS_HPP_
#define _PREPROCESS_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <engine_interface.hpp>

static bool preprocess(tflite::Interpreter * interpreter, int model_mode, uint8_t * rgb_buf_ptr, int img_height, int img_width, int cur_batch = 0){
    for(int i = 0; i < interpreter->inputs().size(); i++){
        TfLiteTensor* input_tensor_i = interpreter->input_tensor(i);
        TfLiteIntArray* input_dims = input_tensor_i->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];
        int input_channel = input_dims->data[3];

        // Calculate scale
        float scale_height = (float) input_height / img_height;
        float scale_width = (float) input_width / img_width;

        switch(model_mode){
            case 1: //ssd_mobilenet
            case 7: //detr_resnet
            case 8: //rt detr
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

        cv::Mat cvImg(cv::Size(img_width, img_height), CV_8UC3, rgb_buf_ptr);

        cv::Mat resizecvImg;
        cv::resize(cvImg, resizecvImg, cv::Size(resize_width, resize_height));
        uint8_t * resize_img_ptr = resizecvImg.data;
        
        // Normalize and pad input
        switch(model_mode){
            case 1: //ssd_mobilenet
            {
                float mean_rgb[3] = {127.5, 127.5, 127.5};
                float stddev_rgb[3] = {127.5, 127.5, 127.5};

                if(interpreter->input_tensor(i)->type == kTfLiteUInt8){
                    uint8_t * input_img_ptr = interpreter->typed_input_tensor<uint8_t>(i, cur_batch);
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx];
                            input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1];
                            input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2];
                        }
                    }
                }
                else if(interpreter->input_tensor(i)->type == kTfLiteFloat32){
                    float * input_img_ptr = interpreter->typed_input_tensor<float>(i, cur_batch);
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = (resize_img_ptr[3 * resize_idx] - mean_rgb[0]) / stddev_rgb[0];
                            input_img_ptr[3 * input_idx + 1] = (resize_img_ptr[3 * resize_idx + 1] - mean_rgb[1]) / stddev_rgb[1];
                            input_img_ptr[3 * input_idx + 2] = (resize_img_ptr[3 * resize_idx + 2] - mean_rgb[2]) / stddev_rgb[2];
                        }
                    }
                }
                else{
                    std::cerr << "ERROR: Unsupported input tensor data type in ssd_mobilenet preprocess\n";
                    return false;
                }
                
                break;
            }
            case 2: //efficientdet
            case 7: //detr resnet
            {
                float mean_rgb[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
                float stddev_rgb[3] = {0.229 * 255, 0.224 * 255, 0.225 * 255};

                if(interpreter->input_tensor(i)->type == kTfLiteUInt8){
                    uint8_t * input_img_ptr = interpreter->typed_input_tensor<uint8_t>(i, cur_batch);
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;
                            
                            input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx];
                            input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1];
                            input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2];
                        }
                    }
                }
                else if(interpreter->input_tensor(i)->type == kTfLiteFloat32){
                    float * input_img_ptr = interpreter->typed_input_tensor<float>(i, cur_batch);
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = (resize_img_ptr[3 * resize_idx] - mean_rgb[0]) / stddev_rgb[0];
                            input_img_ptr[3 * input_idx + 1] = (resize_img_ptr[3 * resize_idx + 1] - mean_rgb[1]) / stddev_rgb[1];
                            input_img_ptr[3 * input_idx + 2] = (resize_img_ptr[3 * resize_idx + 2] - mean_rgb[2]) / stddev_rgb[2];
                        }
                    }
                }
                else{
                    std::cerr << "ERROR: Unsupported input tensor data type in efficientdet/detr resnet preprocess\n";
                    return false;
                }

                break;
            }
            case 3: //efficientdet lite
            {
                float mean_rgb[3] = {127.0, 127.0, 127.0};
                float stddev_rgb[3] = {128.0, 128.0, 128.0};

                if(interpreter->input_tensor(i)->type == kTfLiteUInt8){
                    uint8_t * input_img_ptr = interpreter->typed_input_tensor<uint8_t>(i, cur_batch);
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx];
                            input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1];
                            input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2];
                        }
                    }
                }
                else if(interpreter->input_tensor(i)->type == kTfLiteFloat32){
                    float * input_img_ptr = interpreter->typed_input_tensor<float>(i, cur_batch);
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = (resize_img_ptr[3 * resize_idx] - mean_rgb[0]) / stddev_rgb[0];
                            input_img_ptr[3 * input_idx + 1] = (resize_img_ptr[3 * resize_idx + 1] - mean_rgb[1]) / stddev_rgb[1];
                            input_img_ptr[3 * input_idx + 2] = (resize_img_ptr[3 * resize_idx + 2] - mean_rgb[2]) / stddev_rgb[2];
                        }
                    }
                }
                else{
                    std::cerr << "ERROR: Unsupported input tensor data type in efficientdet lite preprocess\n";
                    return false;
                }
                
                break;
            }
            case 4: //yolo
            case 5: //yolov10
            case 6: //yolo obb
            case 8: //rt detr
            {
                float mean_rgb[3] = {0.0, 0.0, 0.0};
                float stddev_rgb[3] = {255.0, 255.0, 255.0};

                if(interpreter->input_tensor(i)->type == kTfLiteUInt8){
                    uint8_t * input_img_ptr = interpreter->typed_input_tensor<uint8_t>(i, cur_batch);
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(uint8_t));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = resize_img_ptr[3 * resize_idx];
                            input_img_ptr[3 * input_idx + 1] = resize_img_ptr[3 * resize_idx + 1];
                            input_img_ptr[3 * input_idx + 2] = resize_img_ptr[3 * resize_idx + 2];
                        }
                    }
                }
                else if(interpreter->input_tensor(i)->type == kTfLiteFloat32){
                    float * input_img_ptr = interpreter->typed_input_tensor<float>(i, cur_batch);
                    memset(input_img_ptr, 0, input_height * input_width * input_channel * sizeof(float));

                    for(int j = 0; j < resize_height; j++){
                        for(int k = 0; k < resize_width; k++){
                            int input_idx = j * input_width + k;
                            int resize_idx = j * resize_width + k;

                            input_img_ptr[3 * input_idx] = (resize_img_ptr[3 * resize_idx] - mean_rgb[0]) / stddev_rgb[0];
                            input_img_ptr[3 * input_idx + 1] = (resize_img_ptr[3 * resize_idx + 1] - mean_rgb[1]) / stddev_rgb[1];
                            input_img_ptr[3 * input_idx + 2] = (resize_img_ptr[3 * resize_idx + 2] - mean_rgb[2]) / stddev_rgb[2];
                        }
                    }
                }
                else{
                    std::cerr << "ERROR: Unsupported input tensor data type in yolo/yolov10/yolo obb preprocess\n";
                    return false;
                }

                break;
            }
            default:
                std::cerr << "ERROR: Unsupported model mode in preprocess " << model_mode << "\n";
                return false;
        }
    }

    return true;
}

#endif  // _PREPROCESS_HPP_