#include <iostream>
#include <cstddef>
#include <cstdio>
#include <unistd.h>
#include <chrono>
#include <string>
#include <vector>

#include <engine_interface.hpp>
#include "hailo_post.hpp"
#include "maccel_post.hpp"
#include "tflite_post.hpp"
#include "preprocess.hpp"

#include <opencv2/opencv.hpp>

#if defined(__aarch64__)

#include <fastcv/fastcv.h>
#include <qcarcam.h>
#include <qcarcam_client.h>


static tflite::Interpreter * interpreter;
static int model_mode;
static std::vector<std::string> * labels_ptr;

static unsigned long average_inference_time = 0;
static unsigned int num_inference = 0;

static float score_threshold = 0.4;

static int input_height;
static int input_width;
static int input_channel;


// Callback function called when the camera frame is refreshed
static void qcarcam_event_handler(int input_id, unsigned char* buf_ptr, size_t buf_len){
    std::vector<std::string> & labels = *labels_ptr;

    // Get the camera info
    unsigned int queryNumInputs = 0, queryFilled = 0;
    qcarcam_input_t * pInputs;

    if(qcarcam_query_inputs(NULL, 0, &queryNumInputs) != QCARCAM_RET_OK || queryNumInputs == 0){
        std::cerr << "ERROR: The camera is not found.\n";
        exit(-1);
    }

    if(queryNumInputs <= input_id){
        std::cerr << "ERROR: The number of available cameras is smaller than the camera id. Check that all the cameras are connected in order.\n";
        exit(-1);
    }

    pInputs = (qcarcam_input_t *)calloc(queryNumInputs, sizeof(*pInputs));       
    if(!pInputs){
        std::cerr << "ERROR: Failed to calloc\n";
        exit(-1);
    }

    if(qcarcam_query_inputs(pInputs, queryNumInputs, &queryFilled) != QCARCAM_RET_OK || queryFilled != queryNumInputs){
        std::cerr << "ERROR: Failed to get the camera info\n";
        exit(-1);
    }

    int camera_height = pInputs[input_id].res[0].height;
    int camera_width = pInputs[input_id].res[0].width;

    free(pInputs);

    // Change color format from uyuv to rgb
    uint8_t * uv = (uint8_t *)fcvMemAlloc(camera_width * camera_height, 16);
    uint8_t * y = (uint8_t *)fcvMemAlloc(camera_width * camera_height, 16);
    if(uv == NULL || y == NULL){
        std::cerr << "ERROR: Failed to fcvMemAlloc\n";
        exit(-1);
    }

    uint8_t * rgb_buf_ptr = new uint8_t[camera_height * camera_width * 3];
    if(rgb_buf_ptr == NULL){
        std::cerr << "ERROR: Failed memory allocation\n";
        exit(-1);
    }

    fcvDeinterleaveu8(buf_ptr, camera_width, camera_height, camera_width * 2, (uint8_t *)uv, camera_width, (uint8_t *)y, camera_width);
    fcvColorYCbCr422PseudoPlanarToRGB888u8((uint8_t *)y, (uint8_t *)uv, camera_width, camera_height, camera_width, camera_width, (uint8_t *)rgb_buf_ptr, camera_width * 3);

    preprocess(interpreter, model_mode, rgb_buf_ptr, camera_height, camera_width);

    // Inference
    auto start = std::chrono::steady_clock::now();
    if(interpreter->Invoke() != kTfLiteOk){
        std::cerr << "ERROR: Model execute failed\n";
        exit(-1);
    }
    auto elapsed = std::chrono::steady_clock::now() - start;

    average_inference_time = (average_inference_time * num_inference + std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()) / (num_inference + 1);
    num_inference++;

    // Change color format from rgb to bgr for opencv image
    cv::Mat cvimg(camera_height, camera_width, CV_8UC3, rgb_buf_ptr);
    cv::cvtColor(cvimg, cvimg, cv::COLOR_RGB2BGR);

    // Postprocess
    if(interpreter->is_hailo_output()){
        // Get the input tensor size info
        TfLiteTensor* input_tensor_0 = interpreter->hailo_input_tensor(0);
        TfLiteIntArray* input_dims = input_tensor_0->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];

        // Calculate scale
        float scale_height = (float) input_height / camera_height;
        float scale_width = (float) input_width / camera_width;

        switch(model_mode){
            case 1: //ssd_mobilenet
            case 7: //detr resnet
                break;
            default:
            {
                float scale = scale_height < scale_width ? scale_height : scale_width;
                scale_height = scale;
                scale_width = scale;
            }
        }

        switch(model_mode){
            case 1: //ssd mobilenet
            case 3: //efficientdet lite
            case 7: //detr resnet
            {
                std::vector<DetResult> results;
                hailo_post(interpreter, model_mode, results);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    if(score < score_threshold)
                        continue;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    int id = results[i].id - 1;

                    //std::cout << i <<  ": output_classes: " << id << ", output_scores: " << score << ", output_locations: [" << xmin << "," << ymin << "," << xmax << ","<< ymax << "]\n";

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(255, 0, 0), 2);
                }

                break;
            }
            case 4: //yolo
            case 5: //yolov10
            {
                std::vector<DetResult> results;
                hailo_post(interpreter, model_mode, results);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    if(score < score_threshold)
                        continue;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    int id = results[i].id - 1;

                    if(id < 11)
                        id = id;
                    else if(id < 24)
                        id = id + 1;
                    else if(id < 26)
                        id = id + 2;
                    else if(id < 40)
                        id = id + 4;
                    else if(id < 60)
                        id = id + 5;
                    else if(id < 61)
                        id = id + 6;
                    else if(id < 62)
                        id = id + 8;
                    else if(id < 73)
                        id = id + 9;
                    else
                        id = id + 10;

                    //std::cout << i <<  ": output_classes: " << id << ", output_scores: " << score << ", output_locations: [" << xmin << "," << ymin << "," << xmax << ","<< ymax << "]\n";

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(255, 0, 0), 2);
                }

                break;
            }
            default:
            {
                std::cerr << "ERROR: Unsupported model mode in Hailo postprocess " << model_mode << "\n";
                return;
            }
        }
    }
    else if(interpreter->is_maccel_output()){
        // Get the input tensor size info
        TfLiteTensor* input_tensor_0 = interpreter->maccel_input_tensor(0);
        TfLiteIntArray* input_dims = input_tensor_0->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];

        // Calculate scale
        float scale_height = (float) input_height / camera_height;
        float scale_width = (float) input_width / camera_width;

        switch(model_mode){
            case 1: case 7: break;
            default:
            {
                float scale = scale_height < scale_width ? scale_height : scale_width;
                scale_height = scale;
                scale_width = scale;
            }
        }

        switch(model_mode){
            case 4: //yolo
            case 5: //yolov10
            {
                std::vector<DetResult> results;
                maccel_post(interpreter, model_mode, results, false);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    if(score < score_threshold)
                        continue;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    int id = results[i].id;

                    float width = xmax - xmin;
                    float height = ymax - ymin;

                    if(id < 11)
                        id = id;
                    else if(id < 24)
                        id = id + 1;
                    else if(id < 26)
                        id = id + 2;
                    else if(id < 40)
                        id = id + 4;
                    else if(id < 60)
                        id = id + 5;
                    else if(id < 61)
                        id = id + 6;
                    else if(id < 62)
                        id = id + 8;
                    else if(id < 73)
                        id = id + 9;
                    else
                        id = id + 10;

                    //std::cout << i <<  ": output_classes: " << id << ", output_scores: " << score << ", output_locations: [" << xmin << "," << ymin << "," << xmax << ","<< ymax << "]\n";

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(255, 0, 0), 2);
                }

                break;
            }
            case 7: //detr resnet
            {
                std::vector<DetResult> results;
                maccel_post(interpreter, model_mode, results);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    if(score < score_threshold)
                        continue;

                    int id = results[i].id;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    float width = xmax - xmin;
                    float height = ymax - ymin;

                    //std::cout << i <<  ": output_classes: " << id << ", output_scores: " << score << ", output_locations: [" << xmin << "," << ymin << "," << xmax << ","<< ymax << "]\n";

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(255, 0, 0), 2);
                }
                
                break;
            }
            default:
            {
                std::cerr << "ERROR: Unsupported model mode in Maccel postprocess " << model_mode << "\n";
                return;
            }
        }
    }
    else if(interpreter->is_tflite_output()){
        // Get the input tensor size info
        TfLiteTensor* input_tensor_0 = interpreter->tflite_input_tensor(0);
        TfLiteIntArray* input_dims = input_tensor_0->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];

        // Calculate scale
        float scale_height = (float) input_height / camera_height;
        float scale_width = (float) input_width / camera_width;

        switch(model_mode){
            case 1: //ssd_mobilenet
            case 7: //detr resnet
                break;
            default:
            {
                float scale = scale_height < scale_width ? scale_height : scale_width;

                scale_height = scale;
                scale_width = scale;
            }
        }

        switch(model_mode){
            case 1: //ssd mobilenet
            case 2: //efficientdet
            case 3: //efficientdet lite
            case 7: //detr resnet
            {
                std::vector<DetResult> results;
                tflite_post(interpreter, model_mode, results);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    int id = results[i].id;

                    if(score < score_threshold)
                        continue;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(255, 0, 0), 2);
                }

                break;
            }
            case 4: //yolo
            case 5: //yolov10
            {
                std::vector<DetResult> results;
                tflite_post(interpreter, model_mode, results);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    int id = results[i].id;

                    if(score < score_threshold)
                        continue;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    if(id < 11)
                        id = id;
                    else if(id < 24)
                        id = id + 1;
                    else if(id < 26)
                        id = id + 2;
                    else if(id < 40)
                        id = id + 4;
                    else if(id < 60)
                        id = id + 5;
                    else if(id < 61)
                        id = id + 6;
                    else if(id < 62)
                        id = id + 8;
                    else if(id < 73)
                        id = id + 9;
                    else
                        id = id + 10;

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(255, 0, 0), 2);
                }

                break;
            }
            case 6: // yolo obb
            {
                // Get the output tensor size info
                TfLiteTensor *output_tensor_0 = interpreter->tflite_output_tensor(0);
                TfLiteIntArray *output_dims = output_tensor_0->dims;
                int output_height = output_dims->data[1];
                int output_width = output_dims->data[2];

                // Parse output and apply nms
                float *output = interpreter->typed_tflite_output_tensor<float>(0);
                float(*output_arr)[output_width] = (float(*)[output_width])output;

                int max_detections = 300;
                float conf_threshold = score_threshold;
                float iou_threshold = 0.7;
                bool multi_label = false;   // If true, nms is done per class. slow but accurate.

                float pi = 3.14159265358979323846;

                std::vector<int> class_ids;
                std::vector<float> scores;
                std::vector<cv::RotatedRect> boxes;

                for(int i = 0; i < output_width; i++){
                    float cx = output_arr[0][i] * input_width;                  // center x
                    float cy = output_arr[1][i] * input_height;                 // center y
                    float width = output_arr[2][i] * input_width;               // width
                    float height = output_arr[3][i] * input_height;             // height
                    float angle = output_arr[output_height - 1][i] * 180 / pi;  // angle

                    float max_score = 0;
                    int max_class_id = 0;

                    for(int j = 4; j < output_height - 1; j++){
                        float score = output_arr[j][i];

                        int class_id = j - 4;

                        if (score > max_score) {
                            max_score = score;
                            max_class_id = class_id;
                        }

                        if (multi_label) {
                            if (score > conf_threshold) {
                                scores.push_back(score);
                                class_ids.push_back(class_id);
                                cv::RotatedRect rotatedRect(cv::Point2f(cx, cy), cv::Size2f(width, height), angle);
                                boxes.push_back(rotatedRect);
                                //std::cout << i << ", "<< j <<  ": class_id: " << class_id << ", score: " << score << ", cx: " << cx << ", cy: " << cy << ", width: " << width << ", height: "<< height << ", angle: " << angle << std::endl;
                            }
                        }
                    }

                    //std::cout << i << ": max_class_id: " << max_class_id << ", max_score: " << max_score << ", cx: " << cx << ", cy: " << cy << ", width: " << width << ", height: "<< height << ", angle: " << angle << std::endl;

                    if (max_score == 0 && cx == 0 && cy == 0 && width == 0 && height == 0 && angle == 0)
                        break;

                    if (!multi_label) {
                        if (max_score > conf_threshold) {
                            scores.push_back(max_score);
                            class_ids.push_back(max_class_id);
                            cv::RotatedRect rotatedRect(cv::Point2f(cx, cy), cv::Size2f(width, height), angle);
                            boxes.push_back(rotatedRect);
                        }
                    }
                }

                std::vector<int> nms_result;
                if (multi_label) {
                    // Batched nms trick since batched nms function is only available opencv > 4.7.0
                    std::vector<cv::RotatedRect> _boxes = boxes;

                    for (int i = 0; i < _boxes.size(); i++) {
                        cv::Point2f offset(class_ids[i] * input_width, class_ids[i] * input_height);
                        _boxes[i].center = _boxes[i].center + offset;
                    }

                    cv::dnn::NMSBoxes(_boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);
                }
                else {
                    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);
                }

                for (int i = 0; i < nms_result.size(); i++) {
                    int idx = nms_result[i];

                    //std::cout << i <<  ": class_id: " << class_ids[idx] << ", score: " << scores[idx] << ", cx: " << boxes[idx].center.x << ", cy: " << boxes[idx].center.y << ", width: " << boxes[idx].size.height << ", height: "<< boxes[idx].size.width << std::endl;

                    int id = class_ids[idx];
                    float score = scores[idx];

                    cv::Point2f center = boxes[idx].center;
                    cv::Size2f size = boxes[idx].size;
                    float angle = boxes[idx].angle;

                    float cx = center.x / scale_width;
                    float cy = center.y / scale_height;
                    float width = size.width / scale_width;
                    float height = size.height / scale_height;

                    float xmin = cx - 0.5 * width;
                    float xmax = cx + 0.5 * width;
                    float ymin = cy - 0.5 * height;
                    float ymax = cy + 0.5 * height;

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0));

                    cv::RotatedRect rotatedRect(cv::Point2f(cx, cy), cv::Size2f(width, height), angle);
                    cv::Point2f vertices[4];
                    rotatedRect.points(vertices);

                    for (int i = 0; i < 4; i++)
                        line(cvimg, vertices[i], vertices[(i+1)%4], cv::Scalar(255, 0, 0), 2);
                }

                break;
            }
            default:
            {
                std::cerr << "ERROR: Unsupported model mode in TFLite postprocess " << model_mode << "\n";
                return;
            }
        }
    }

    memcpy(rgb_buf_ptr, cvimg.data, camera_width * camera_height * 3 * sizeof(uint8_t));

    // Change color format from bgr to uyuv
    fcvColorRGB888ToYCbCr422PseudoPlanaru8(rgb_buf_ptr, camera_width, camera_height, camera_width * 3, y, uv, camera_width, camera_width);
    fcvInterleaveu8(uv, y, camera_width, camera_height, camera_width, camera_width, buf_ptr, camera_width * 2);

    // Free memory
    delete[] rgb_buf_ptr;
    fcvMemFree(uv);
    fcvMemFree(y);
}

bool run_qcarcam(tflite::Interpreter * interpreter_arg, int model_mode_arg, std::vector<std::string> * labels_arg, char * display_path){
    interpreter = interpreter_arg;
    model_mode = model_mode_arg;
    labels_ptr = labels_arg;

    // Get the input tensor size info
    TfLiteTensor* input_tensor_0 = interpreter->input_tensor(0);
    TfLiteIntArray* input_dims = input_tensor_0->dims;
    input_height = input_dims->data[1];
    input_width = input_dims->data[2];
    input_channel = input_dims->data[3];

    // Run qcarcam
    if(qcarcam_client_start_preview(display_path, qcarcam_event_handler) != QCARCAM_RET_OK){
        std::cerr << "ERROR: Cannot connect to the qcarcam. Please check the display setting file.\n";
        exit(-1);
    }

    // Wait the exit
    std::cout << "\nPress ctrl+c to exit.\n\n";
    int secs = 0;
    while (true){
        sleep(10);
        secs += 10;
        std::cout << std::fixed;
        std::cout.precision(3);
        std::cout << "Average inference speed(0~" << secs << "s): " << 1000.0 / average_inference_time << "fps\n";
        std::cout << "Average inference speed(0~" << secs << "s): " << average_inference_time << "ms\n\n";
    }

    // Stop qcarcam
    if(qcarcam_client_stop_preview() != QCARCAM_RET_OK){
        std::cerr << "ERROR: Cannot disconnect the qcarcam.\n";
        exit(-1);
    }

    return true;
}

#else
bool run_qcarcam(tflite::Interpreter * interpreter_arg, int model_mode_arg, std::vector<std::string> * labels_arg, char * display_path){
    (void)interpreter_arg;
    (void)model_mode_arg;
    (void)labels_arg;
    (void)display_path;

    std::cerr << "ERROR: Camera mode (qcarcam) is not supported on this platform (non-ARM64)." << std::endl;
    return false;
}
#endif // defined(__aarch64__)