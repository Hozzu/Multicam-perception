#include <iostream>
#include <fstream>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <chrono>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <time.h>

#include <pthread.h>

#include <engine_interface.hpp>
#include "hailo_post.hpp"
#include "maccel_post.hpp"
#include "tflite_post.hpp"
#include "preprocess.hpp"

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <jpeglib.h>
#include <json-c/json.h>

// --- Global Statistics ---
static float score_threshold = 0.001;

static unsigned int num_preproces = 0;
static unsigned int num_postprocess = 0;
static unsigned int num_infer = 0;
static double sum_preprocess_time = 0;
static double sum_postprocess_time = 0;
static double sum_infer_time = 0;

static unsigned int num_turnaround = 0;
static double sum_turnaround = 0;
static double max_turnaround = 0;

static std::mutex in_postprocess_mutex;
static std::mutex in_preprocess_mutex;

static double get_thread_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
}

// --- Thread Pool Implementation ---
class ThreadPool {
public:
    ThreadPool(size_t num_threads, const std::vector<int>& cpus) : stop(false), active_tasks(0) {
        for(size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this, cpus] {
                // Set Affinity once per worker thread
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                for(int cpu : cpus) {
                    CPU_SET(cpu, &cpuset);
                }
                pthread_t current_thread = pthread_self();
                if(pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
                    std::cerr << "ERROR: Failed to set thread affinity in ThreadPool.\n";
                }

                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                    
                    // Task finished
                    std::unique_lock<std::mutex> lock(this->wait_mutex);
                    active_tasks--;
                    wait_condition.notify_all();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
            // Increment active tasks count
            std::unique_lock<std::mutex> wait_lock(wait_mutex);
            active_tasks++;
        }
        condition.notify_one();
    }

    // Wait until all tasks in the queue are finished
    void wait_all() {
        std::unique_lock<std::mutex> lock(wait_mutex);
        wait_condition.wait(lock, [this]() {
            std::unique_lock<std::mutex> q_lock(queue_mutex);
            return tasks.empty() && active_tasks == 0;
        });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    // Synchronization for wait_all
    std::mutex wait_mutex;
    std::condition_variable wait_condition;
    int active_tasks;
};


// --- Task Functions (Modified to run inside Pool) ---
static void preprocess_task(tflite::Interpreter * interpreter, int model_mode, std::string filename, json_object * json_images, int cur_batch, std::vector<int> &img_heights, std::vector<int> &img_widths, int image_id){ 
    // Decode the image
    in_preprocess_mutex.lock();
    std::cout << "Detecting " << filename << "..\r";
    std::cout.flush();
    in_preprocess_mutex.unlock();

    double preprocess_start = get_thread_time_ms();

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    FILE * fp = fopen(filename.c_str(), "rb");
    if(fp == NULL) {
        std::cerr << "ERROR: Cannot open the image: " << filename << std::endl;
        jpeg_destroy_decompress(&cinfo);
        return; 
    }
    jpeg_stdio_src(&cinfo, fp);

    jpeg_read_header(&cinfo, TRUE);

    cinfo.out_color_space = JCS_RGB;
    cinfo.output_components = 3;

    jpeg_start_decompress(&cinfo);

    // Get the image data
    int img_height = cinfo.output_height;
    int img_width = cinfo.output_width;
    int row_stride = cinfo.output_width * 3;

    // Direct access to vector is safe because each task accesses a unique cur_batch index
    img_heights[cur_batch] = img_height;
    img_widths[cur_batch] = img_width;

    uint8_t * rgb_buf_ptr = (uint8_t *)malloc(sizeof(uint8_t) * img_height * img_width * 3);
    while(cinfo.output_scanline < cinfo.output_height){
        uint8_t * rowptr = rgb_buf_ptr + row_stride * cinfo.output_scanline; 
        jpeg_read_scanlines(&cinfo, &rowptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    fclose(fp);

    // Write json images
    json_object * json_image = json_object_new_object();

    json_object_object_add(json_image, "id", json_object_new_int(image_id));
    json_object_object_add(json_image, "file_name", json_object_new_string(filename.c_str()));
    json_object_object_add(json_image, "width", json_object_new_int(img_width));
    json_object_object_add(json_image, "height", json_object_new_int(img_height));

    in_preprocess_mutex.lock();
    json_object_array_add(json_images, json_image);
    in_preprocess_mutex.unlock();

    preprocess(interpreter, model_mode, rgb_buf_ptr, img_height, img_width, cur_batch);

    free(rgb_buf_ptr);

    double preprocess_elapsed = get_thread_time_ms() - preprocess_start;
    in_preprocess_mutex.lock();
    sum_preprocess_time += preprocess_elapsed;
    num_preproces++;
    in_preprocess_mutex.unlock();
}

static void postprocess_task(tflite::Interpreter * interpreter, int model_mode, std::vector<int> &img_heights, std::vector<int> &img_widths, std::vector<int> &image_ids, json_object * json_annotations, int cur_batch){
    double postprocess_start = get_thread_time_ms();

    // Postprocess
    if(interpreter->is_tflite_output(cur_batch)){
        // Get the input tensor size info
        TfLiteTensor* input_tensor_0 = interpreter->tflite_input_tensor(0);
        TfLiteIntArray* input_dims = input_tensor_0->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];

        // Calculate scale
        float scale_height = (float) input_height / img_heights[cur_batch];
        float scale_width = (float) input_width / img_widths[cur_batch];

        switch(model_mode){
            case 1: //ssd_mobilenet
            case 7: //detr resnet
            case 8: //rt detr
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
                tflite_post(interpreter, model_mode, results, true, cur_batch);

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

                    // Write json annotations
                    json_object * json_annotation = json_object_new_object();
                    json_object * json_bbox = json_object_new_array();

                    json_object_array_add(json_bbox, json_object_new_int(round(xmin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(ymin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(width)));
                    json_object_array_add(json_bbox, json_object_new_int(round(height)));

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "bbox", json_bbox);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));
                    
                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
                }

                break;
            }
            case 4: //yolo
            case 5: //yolov10
            case 8: //rt-detr
            {
                std::vector<DetResult> results;
                tflite_post(interpreter, model_mode, results, true, cur_batch);

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

                    if(id < 11) id = id;
                    else if(id < 24) id = id + 1;
                    else if(id < 26) id = id + 2;
                    else if(id < 40) id = id + 4;
                    else if(id < 60) id = id + 5;
                    else if(id < 61) id = id + 6;
                    else if(id < 62) id = id + 8;
                    else if(id < 73) id = id + 9;
                    else id = id + 10;

                    json_object * json_annotation = json_object_new_object();
                    json_object * json_bbox = json_object_new_array();

                    json_object_array_add(json_bbox, json_object_new_int(round(xmin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(ymin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(width)));
                    json_object_array_add(json_bbox, json_object_new_int(round(height)));

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "bbox", json_bbox);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));

                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
                }
                break;
            }
            case 6: //yolo obb
            {
                TfLiteTensor *output_tensor_0 = interpreter->tflite_output_tensor(0);
                TfLiteIntArray *output_dims = output_tensor_0->dims;
                int output_height = output_dims->data[1];
                int output_width = output_dims->data[2];

                float *output = interpreter->typed_tflite_output_tensor<float>(0, cur_batch);
                float(*output_arr)[output_width] = (float(*)[output_width])output;

                int max_detections = 300;
                float conf_threshold = 0.001;
                float iou_threshold = 0.7;
                bool multi_label = true;    // If true, nms is done per class. slow but accurate.
                float pi = 3.14159265358979323846;

                std::vector<int> class_ids;
                std::vector<float> scores;
                std::vector<cv::RotatedRect> boxes;

                for(int i = 0; i < output_width; i++){
                    float cx = output_arr[0][i] * input_width;
                    float cy = output_arr[1][i] * input_height;
                    float width = output_arr[2][i] * input_width;
                    float height = output_arr[3][i] * input_height;
                    float angle = output_arr[output_height - 1][i] * 180 / pi;

                    float max_score = 0;
                    int max_class_id = 0;

                    for(int j = 4; j < output_height - 1; j++){
                        float score = output_arr[j][i];
                        int class_id = j - 4;
                        if (score > max_score) { max_score = score; max_class_id = class_id; }
                        if (multi_label && score > conf_threshold) {
                            scores.push_back(score); class_ids.push_back(class_id);
                            boxes.push_back(cv::RotatedRect(cv::Point2f(cx, cy), cv::Size2f(width, height), angle));
                        }
                    }
                    if (!multi_label && max_score > conf_threshold) {
                        scores.push_back(max_score); class_ids.push_back(max_class_id);
                        boxes.push_back(cv::RotatedRect(cv::Point2f(cx, cy), cv::Size2f(width, height), angle));
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
                } else {
                    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);
                }

                for (int i = 0; i < nms_result.size(); i++) {
                    int idx = nms_result[i];
                    int id = class_ids[idx];
                    float score = scores[idx];

                    cv::Point2f center = boxes[idx].center;
                    cv::Size2f size = boxes[idx].size;
                    float angle = boxes[idx].angle;
                    float cx = center.x / scale_width;
                    float cy = center.y / scale_height;
                    float width = size.width / scale_width;
                    float height = size.height / scale_height;

                    cv::RotatedRect rotatedRect(cv::Point2f(cx, cy), cv::Size2f(width, height), angle);
                    cv::Point2f vertices[4];
                    rotatedRect.points(vertices);

                    json_object *json_annotation = json_object_new_object();
                    json_object *rbox = json_object_new_array();
                    json_object *poly = json_object_new_array();

                    json_object_array_add(rbox, json_object_new_double(cx));
                    json_object_array_add(rbox, json_object_new_double(cy));
                    json_object_array_add(rbox, json_object_new_double(width));
                    json_object_array_add(rbox, json_object_new_double(height));
                    json_object_array_add(rbox, json_object_new_double(angle));
                    
                    for(int v=0; v<4; v++) {
                        json_object_array_add(poly, json_object_new_double(vertices[v].x));
                        json_object_array_add(poly, json_object_new_double(vertices[v].y));
                    }

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "rbox", rbox);
                    json_object_object_add(json_annotation, "poly", poly);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));

                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
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
    else if(interpreter->is_hailo_output(cur_batch)){
        // Get the input tensor size info
        TfLiteTensor* input_tensor_0 = interpreter->hailo_input_tensor(0);
        TfLiteIntArray* input_dims = input_tensor_0->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];

        // Calculate scale
        float scale_height = (float) input_height / img_heights[cur_batch];
        float scale_width = (float) input_width / img_widths[cur_batch];

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
                hailo_post(interpreter, model_mode, results, cur_batch);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    if(score < score_threshold) continue;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    int id = results[i].id;

                    float width = xmax - xmin;
                    float height = ymax - ymin;

                    json_object * json_annotation = json_object_new_object();
                    json_object * json_bbox = json_object_new_array();

                    json_object_array_add(json_bbox, json_object_new_int(round(xmin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(ymin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(width)));
                    json_object_array_add(json_bbox, json_object_new_int(round(height)));

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "bbox", json_bbox);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));

                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
                }
                break;
            }
            case 4: //yolo
            case 5: //yolov10
            {
                std::vector<DetResult> results;
                hailo_post(interpreter, model_mode, results, cur_batch);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    if(score < score_threshold) continue;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    int id = results[i].id;

                    if(id < 11) id = id;
                    else if(id < 24) id = id + 1;
                    else if(id < 26) id = id + 2;
                    else if(id < 40) id = id + 4;
                    else if(id < 60) id = id + 5;
                    else if(id < 61) id = id + 6;
                    else if(id < 62) id = id + 8;
                    else if(id < 73) id = id + 9;
                    else id = id + 10;

                    float width = xmax - xmin;
                    float height = ymax - ymin;

                    json_object * json_annotation = json_object_new_object();
                    json_object * json_bbox = json_object_new_array();

                    json_object_array_add(json_bbox, json_object_new_int(round(xmin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(ymin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(width)));
                    json_object_array_add(json_bbox, json_object_new_int(round(height)));

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "bbox", json_bbox);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));

                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
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
    else if(interpreter->is_maccel_output(cur_batch)){
        // Get the input tensor size info
        TfLiteTensor* input_tensor_0 = interpreter->maccel_input_tensor(0);
        TfLiteIntArray* input_dims = input_tensor_0->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];

        // Calculate scale
        float scale_height = (float) input_height / img_heights[cur_batch];
        float scale_width = (float) input_width / img_widths[cur_batch];

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
                maccel_post(interpreter, model_mode, results, true, cur_batch);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    if(score < score_threshold) continue;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    int id = results[i].id;

                    float width = xmax - xmin;
                    float height = ymax - ymin;

                    if(id < 11) id = id;
                    else if(id < 24) id = id + 1;
                    else if(id < 26) id = id + 2;
                    else if(id < 40) id = id + 4;
                    else if(id < 60) id = id + 5;
                    else if(id < 61) id = id + 6;
                    else if(id < 62) id = id + 8;
                    else if(id < 73) id = id + 9;
                    else id = id + 10;

                    json_object * json_annotation = json_object_new_object();
                    json_object * json_bbox = json_object_new_array();

                    json_object_array_add(json_bbox, json_object_new_int(round(xmin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(ymin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(width)));
                    json_object_array_add(json_bbox, json_object_new_int(round(height)));

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "bbox", json_bbox);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));

                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
                }
                break;
            }
            case 7: //detr resnet
            {
                std::vector<DetResult> results;
                maccel_post(interpreter, model_mode, results, true, cur_batch);

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

                    // Write json annotations
                    json_object * json_annotation = json_object_new_object();
                    json_object * json_bbox = json_object_new_array();

                    json_object_array_add(json_bbox, json_object_new_int(round(xmin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(ymin)));
                    json_object_array_add(json_bbox, json_object_new_int(round(width)));
                    json_object_array_add(json_bbox, json_object_new_int(round(height)));

                    json_object_object_add(json_annotation, "image_id", json_object_new_int(image_ids[cur_batch]));
                    json_object_object_add(json_annotation, "bbox", json_bbox);
                    json_object_object_add(json_annotation, "category_id", json_object_new_int(id + 1));
                    json_object_object_add(json_annotation, "score", json_object_new_double(score));
                    
                    in_postprocess_mutex.lock();
                    json_object_array_add(json_annotations, json_annotation);
                    in_postprocess_mutex.unlock();
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

    double postprocess_elapsed = get_thread_time_ms();
    in_postprocess_mutex.lock();
    sum_postprocess_time += postprocess_elapsed;
    num_postprocess++;
    in_postprocess_mutex.unlock();
}


static void infer(tflite::Interpreter * interpreter, int model_mode, char * directory_path, json_object * json_images, json_object * json_annotations, int batch_size){
    // Set Affinity for the Inference Thread (Cores 4+)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < num_cores; i++) {
        if (i != 0 && i != 1 && i != 2 && i != 3) {
            CPU_SET(i, &cpuset);
        }
    }
    if (CPU_COUNT(&cpuset) == 0) {
        std::cerr << "ERROR: No available CPU cores (Total cores <= 4). Cannot exclude 0-3.\n";
        return;
    }
    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "ERROR: Failed to set thread affinity.\n";
        return;
    } else {
        std::cout << "INFO: Affinity set to Cores: ";
        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &cpuset)) {
                printf("%d,", i);
            }
        }
        printf("\n");
    }

    // Initialize Thread Pools
    ThreadPool pre_pool(4, {0, 1, 2, 3});
    ThreadPool post_pool(4, {0, 1, 2, 3});

    DIR * dir = opendir(directory_path);
    if(dir == NULL){
        std::cerr << "ERROR: Cannot open the directory.\n";
        return;
    }

    struct dirent * ent;
    int image_id = 0;

    // Double buffering data structures for Pipelining
    // buffer 0: current, buffer 1: next
    std::vector<int> img_heights[2], img_widths[2], image_ids[2];
    for(int k=0; k<2; k++) {
        img_heights[k].resize(batch_size);
        img_widths[k].resize(batch_size);
        image_ids[k].resize(batch_size);
    }
    int cur_buf_idx = 0;
    int prev_buf_idx = 1;

    bool is_first_batch = true;
    int prev_batch_count = 0;

    while(true){
        int cur_batch_count = 0;
        
        // --- 1. Fill the current batch list ---
        std::vector<std::string> batch_filenames;
        
        while(cur_batch_count < batch_size){
            ent = readdir(dir);
            if(ent == NULL) break;

            const char * filename = ent->d_name;               
            if(strstr(filename, ".jpg")){
                image_id++;
                
                // Set metadata in the CURRENT buffer
                image_ids[cur_buf_idx][cur_batch_count] = image_id;
                batch_filenames.push_back(std::string(filename));
                
                cur_batch_count++;
            }
        }

        if(cur_batch_count == 0) {
            // No more images.
            // If there was a previous batch running postprocess, wait for it.
            if(!is_first_batch) {
                post_pool.wait_all();
                max_turnaround += interpreter->GetMaxTurnAroundTime();
                sum_turnaround += interpreter->GetSumTurnAroundTime();
                num_turnaround++;
            }
            break;
        }

        // --- 2. Schedule Preprocessing for CURRENT batch ---
        // This runs in parallel with step 3 (waiting for previous postprocess)
        for(int i = 0; i < cur_batch_count; i++) {
            pre_pool.enqueue([=, &img_heights, &img_widths, &image_ids]() {
                // IMPORTANT: We must capture the buffer index by value here or ensure it's stable
                preprocess_task(interpreter, model_mode, batch_filenames[i], json_images, i, 
                                std::ref(img_heights[cur_buf_idx]), std::ref(img_widths[cur_buf_idx]), image_ids[cur_buf_idx][i]);
            });
        }

        // --- 3. Wait for PREVIOUS Postprocess to finish ---
        if(!is_first_batch) {
            post_pool.wait_all();
            max_turnaround += interpreter->GetMaxTurnAroundTime();
            sum_turnaround += interpreter->GetSumTurnAroundTime();
            num_turnaround++;
        }

        // --- 4. Wait for CURRENT Preprocess to finish ---
        // We cannot Invoke until preprocessing is done.
        pre_pool.wait_all();

        // --- Invoke ---
        auto invoke_start = std::chrono::steady_clock::now();

        // Invoke
        if(interpreter->Invoke(true) != kTfLiteOk){
            std::cerr << "ERROR: Model execute failed\n";
            break;
        }
// =====================================================================
/*auto print_tensor_dump = [&](int tensor_id, const std::string& name) {
    TfLiteTensor* t = interpreter->tensor(tensor_id);
    if (!t || t->data.data == nullptr) {
        std::cout << "[DUMP] " << name << " (ID: " << tensor_id << ") is null or not available.\n";
        return;
    }
    
    int num_elements = 1;
    for (int i = 0; i < t->dims->size; ++i) {
        num_elements *= t->dims->data[i];
    }

    std::cout << "\n========== " << name << " (ID: " << tensor_id << ") ==========\n";
    
    std::string type_str = "UNKNOWN";
    if (t->type == kTfLiteFloat32) type_str = "FLOAT32";
    else if (t->type == kTfLiteInt8) type_str = "INT8";
    else if (t->type == kTfLiteUInt8) type_str = "UINT8";
    else if (t->type == kTfLiteInt32) type_str = "INT32";

    std::cout << "Type: " << type_str << "\n";
    std::cout << "Quant Params: Scale = " << t->params.scale 
              << ", ZeroPoint = " << t->params.zero_point << "\n";

    std::cout << "Shape:[";
    for (int i = 0; i < t->dims->size; ++i) {
        std::cout << t->dims->data[i] << (i == t->dims->size - 1 ? "" : ", ");
    }
    std::cout << "]\n";

    auto print_val = [&](int index) {
        if (t->type == kTfLiteFloat32) {
            std::cout << t->data.f[index] << " ";
        } else if (t->type == kTfLiteInt8) {
            std::cout << static_cast<int>(t->data.int8[index]) << " ";
        } else if (t->type == kTfLiteUInt8) {
            std::cout << static_cast<int>(t->data.uint8[index]) << " ";
        } else if (t->type == kTfLiteInt32) {
            std::cout << t->data.i32[index] << " ";
        } else {
            std::cout << "? ";
        }
    };

    std::cout << "First 64 values: ";
    for (int i = 0; i < std::min(64, num_elements); ++i) {
        print_val(i);
    }
    
    std::cout << "\nLast 64 values:  ";
    for (int i = std::max(0, num_elements - 64); i < num_elements; ++i) {
        print_val(i);
    }
    std::cout << "\n======================================================\n";
};

//detr resnet 18
print_tensor_dump(310,  "Node 1: PAD");
print_tensor_dump(311,  "Node 2: CONV_2D");
print_tensor_dump(344,  "Node 35: CONV_2D");
print_tensor_dump(355,  "Node 46: FULLY_CONNECTED");
print_tensor_dump(358,  "Node 49: BatchedMatMul");
print_tensor_dump(361,  "Node 52: BatchedMatMul");
print_tensor_dump(364,  "Node 55: FULLY_CONNECTED");
print_tensor_dump(365,  "Node 56: GATHER");
print_tensor_dump(369,  "Node 60: FULLY_CONNECTED");
print_tensor_dump(421,  "Node 112: FULLY_CONNECTED");
print_tensor_dump(500,  "Node 191: FULLY_CONNECTED");
print_tensor_dump(503,  "Node 194: FULLY_CONNECTED");
print_tensor_dump(522,  "Node 213: FULLY_CONNECTED");
print_tensor_dump(523,  "Node 214: RESHAPE");
print_tensor_dump(524,  "Node 215: TRANSPOSE");
print_tensor_dump(525,  "Node 216: BATCHED_MATMUL");
print_tensor_dump(526,  "Node 217: MUL");
print_tensor_dump(527,  "Node 218: SOFTMAX");
print_tensor_dump(528,  "Node 219: BATCHED_MATMUL");
print_tensor_dump(531,  "Node 222: FULLY_CONNECTED");
print_tensor_dump(532,  "Node 223: GATHER");*/

//detr resnet 50
/*print_tensor_dump(379,  "Node 0: Quantize");
print_tensor_dump(381,  "Node 2: Conv2D");
print_tensor_dump(383,  "Node 4: MaxPool2D");
print_tensor_dump(454,  "Node 75: ADD");
print_tensor_dump(455,  "Node 76: Conv2D");
print_tensor_dump(472,  "Node 93: BatchMatMul");
print_tensor_dump(476,  "Node 97: Gather");
print_tensor_dump(477,  "Node 98: ADD");
print_tensor_dump(478,  "Node 99: MEAN");
print_tensor_dump(479,  "Node 100: SUB");
print_tensor_dump(482,  "Node 103: ADD");
print_tensor_dump(483,  "Node 104: Rsqt");
print_tensor_dump(484,  "Node 105: MUL");
print_tensor_dump(488,  "Node 109: ADD");
print_tensor_dump(490,  "Node 111: ADD");
print_tensor_dump(702,  "Node 323: MUL");
print_tensor_dump(704,  "Node 325: ADD");
print_tensor_dump(744,  "Node 365: ADD");
print_tensor_dump(786,  "Node 407: ADD");
print_tensor_dump(790,  "Node 411: FULLY_CONNECTED");
print_tensor_dump(851,  "Node 472: ADD");
print_tensor_dump(867,  "Node 488: ADD");
print_tensor_dump(917,  "Node 538: ADD");
print_tensor_dump(988,  "Node 609: FULLY_CONNECTED");
print_tensor_dump(983,  "Node 604: ADD");
print_tensor_dump(999,  "Node 620: ADD");
print_tensor_dump(1097,  "Node 718: RESHAPE");


//yolov11n
/*print_tensor_dump(471,  "Node 2 : CONV_2D");
print_tensor_dump(473,  "Node 4 : MUL");
print_tensor_dump(475,  "Node 6 : CONV_2D");
print_tensor_dump(480,  "Node 11 : MUL");
print_tensor_dump(493,  "Node 24 : CONCATENATION");
print_tensor_dump(499,  "Node 30 : CONV_2D");
print_tensor_dump(558,  "Node 89 : CONCATENATION");
print_tensor_dump(598,  "Node 129 : CONCATENATION");
print_tensor_dump(622,  "Node 153 : STRIDED_SLICE");
print_tensor_dump(623,  "Node 154 : BATCHED_MATMUL");
print_tensor_dump(624,  "Node 155 : MUL");
print_tensor_dump(625,  "Node 156 : SOFTMAX");
print_tensor_dump(627,  "Node 158 : BATCHED_MATMUL");
print_tensor_dump(628,  "Node 159 : RESHAPE");
print_tensor_dump(887,  "Node 291 : CONCATENATION");
print_tensor_dump(888,  "Node 292 : ADD");
print_tensor_dump(889,  "Node 293 : TRANSPOSE");
print_tensor_dump(890,  "Node 294 : ADD");
print_tensor_dump(891,  "Node 295 : CONV_2D");
print_tensor_dump(892,  "Node 296 : ADD");
print_tensor_dump(893,  "Node 297 : CONV_2D");
print_tensor_dump(894,  "Node 298 : LOGISTIC");
print_tensor_dump(895,  "Node 299 : MUL");
print_tensor_dump(899,  "Node 303 : CONCATENATION");
print_tensor_dump(901,  "Node 305 : CONV_2D");
print_tensor_dump(924,  "Node 328 : CONV_2D");
print_tensor_dump(929,  "Node 333 : CONCATENATION");
print_tensor_dump(930,  "Node 334 : CONV_2D");
print_tensor_dump(945,  "Node 349 : CONCATENATION");
print_tensor_dump(947,  "Node 351 : CONV_2D");
print_tensor_dump(985,  "Node 389 : CONV_2D");
print_tensor_dump(1054,  "Node 458 : CONV_2D");
print_tensor_dump(1062,  "Node 466 : CONCATENATION");
print_tensor_dump(1063,  "Node 467 : TRANSPOSE");
print_tensor_dump(1064,  "Node 468 : RESHAPE");
print_tensor_dump(1072,  "Node 476 : CONCATENATION");
print_tensor_dump(1074,  "Node 478 : RESHAPE");
print_tensor_dump(1075,  "Node 479 : CONV_2D");
print_tensor_dump(1078,  "Node 482 : CONV_2D");
print_tensor_dump(1081,  "Node 485 : CONV_2D");
print_tensor_dump(1082,  "Node 486 : CONCATENATION");
print_tensor_dump(1085,  "Node 489 : CONCATENATION");
print_tensor_dump(1086,  "Node 490 : STRIDED_SLICE");
print_tensor_dump(1087,  "Node 491 : RESHAPE");
print_tensor_dump(1088,  "Node 492 : TRANSPOSE");
print_tensor_dump(1089,  "Node 493 : SOFTMAX");
print_tensor_dump(1090,  "Node 494 : CONV_2D");
print_tensor_dump(1102,  "Node 506 : LOGISTIC");
print_tensor_dump(1100,  "Node 504 : SUB");*/

//yolov12n
/*print_tensor_dump(1081,  "Node 23 : CONCATENATION");
print_tensor_dump(1082,  "Node 24 : Transpose");
print_tensor_dump(1083,  "Node 25 : Conv2D");
print_tensor_dump(1087,  "Node 29 : Conv2D");
print_tensor_dump(1090,  "Node 32 : Conv2D");
print_tensor_dump(1092,  "Node 34 : MUL");
print_tensor_dump(1105,  "Node 47 : CONCATENATION");
print_tensor_dump(1121,  "Node 63 : Transpose");
print_tensor_dump(1125,  "Node 67 : Transpose");
print_tensor_dump(1137,  "Node 72 : Split");
print_tensor_dump(1159,  "Node 87 : Transpose");
print_tensor_dump(1128,  "Node 70 : Split");
print_tensor_dump(1160,  "Node 88 : FullyConnected");
print_tensor_dump(1169,  "Node 97 : PACK");
print_tensor_dump(1170,  "Node 98 : MUL");
print_tensor_dump(1171,  "Node 99 : RESHAPE");
print_tensor_dump(1172,  "Node 100 : SOFTMAX");
print_tensor_dump(1176,  "Node 104 : RESHAPE");
print_tensor_dump(1305,  "Node 170 : CONCATENATION");
print_tensor_dump(1793,  "Node 476 : ADD");
print_tensor_dump(2401,  "Node 846 : ADD");
print_tensor_dump(3395,  "Node 1423 : ADD");
//print_tensor_dump(3636,  "Node 1658 : RESHAPE");
//print_tensor_dump(3646,  "Node 1668 : RESHAPE");
//print_tensor_dump(3656,  "Node 1678 : RESHAPE");
//print_tensor_dump(3657,  "Node 1679 : CONCATENATION");
//print_tensor_dump(3658,  "Node 1680 : STRIDED_SLICE");
//print_tensor_dump(3659,  "Node 1681 : Reshape");
//print_tensor_dump(3660,  "Node 1682 : Transpose");
//print_tensor_dump(3661,  "Node 1683 : SOFTMAX");
//print_tensor_dump(3662,  "Node 1684 : CONV_2D");
//print_tensor_dump(3890,  "Node 1316 : Concatenation");
//print_tensor_dump(3891,  "Output");*/

//yolov8n
/*print_tensor_dump(0,  "INPUT");
print_tensor_dump(159,  "CONV_2D");*/
// =====================================================================
        auto infer_elapsed = std::chrono::steady_clock::now() - invoke_start;
        sum_infer_time += std::chrono::duration_cast<std::chrono::milliseconds>(infer_elapsed).count();
        num_infer++;

        // --- 5. Schedule Postprocessing for CURRENT batch ---
        // Fire and forget (until next loop iteration).
        // Use the current buffer index. The next loop will use the other buffer.
        for(int i = 0; i < cur_batch_count; i++){
            post_pool.enqueue([=, &img_heights, &img_widths, &image_ids](){
                postprocess_task(interpreter, model_mode, 
                                 std::ref(img_heights[cur_buf_idx]), std::ref(img_widths[cur_buf_idx]), std::ref(image_ids[cur_buf_idx]), 
                                 json_annotations, i);
            });
        }

        // --- Prepare for next iteration ---
        is_first_batch = false;
        prev_batch_count = cur_batch_count;
        
        // Swap buffers
        prev_buf_idx = cur_buf_idx;
        cur_buf_idx = 1 - cur_buf_idx;

        if(ent == NULL) {
            // We broke out of the inner loop because of EOF, but we still scheduled the last postprocess.
            // We need to wait for it before exiting the function.
            post_pool.wait_all();
            max_turnaround += interpreter->GetMaxTurnAroundTime();
            sum_turnaround += interpreter->GetSumTurnAroundTime();
            num_turnaround++;
            break; 
        }
    }
    
    // Final cleanup check
    post_pool.wait_all();

    closedir(dir);
}

bool run_image(tflite::Interpreter * interpreter, int model_mode, std::vector<std::string> * labels_arg, char * directory_path, char * result_path, int batch_size = 1){
    // Resize Interpreter Inputs
    for(int i = 0; i < interpreter->inputs().size(); i++){
        TfLiteTensor* input_tensor_i = interpreter->input_tensor(i);
        TfLiteIntArray* input_dims = input_tensor_i->dims;

        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];
        int input_channel = input_dims->data[3];

        int input_tensor_idx = interpreter->inputs()[i];
        if(interpreter->ResizeInputTensor(input_tensor_idx, {batch_size, input_height, input_width, input_channel}) != kTfLiteOk){
            std::cerr << "WARNING: Input resize failed.\n";
            batch_size = 1;
            break;
        }
    }
    
    if(interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "ERROR: Memory allocation for interpreter failed.\n";
        return false;
    }
    
    std::vector<std::string> & labels = *labels_arg;

    auto application_start = std::chrono::steady_clock::now();

    // Create json object
    json_object * json_result = json_object_new_object();
    json_object * json_categories = json_object_new_array();
    json_object * json_images = json_object_new_array();
    json_object * json_annotations = json_object_new_array();

    // Write json categories from label
    for(int i = 0; i < labels.size(); i++){
        if(labels[i] == "???" || labels[i] == "")
            continue;

        json_object * json_category = json_object_new_object();

        json_object_object_add(json_category, "id", json_object_new_int(i + 1));
        json_object_object_add(json_category, "name", json_object_new_string(labels[i].c_str()));

        json_object_array_add(json_categories, json_category);
    }

    char current_dir[500];
    getcwd(current_dir, 500);
    chdir(directory_path);

    infer(interpreter, model_mode, directory_path, json_images, json_annotations, batch_size);

    auto application_elapsed = std::chrono::steady_clock::now() - application_start;
    auto application_latency = std::chrono::duration_cast<std::chrono::milliseconds>(application_elapsed).count();

    std::cout << std::fixed;
    std::cout.precision(3);
    std::cout << std::endl << std::endl;
    std::cout << "Average preprocess time:\t" << sum_preprocess_time / num_preproces << " ms, Num preprocess: " << num_preproces << "\n";
    std::cout << "Average infer time:\t" << sum_infer_time / num_infer << " ms, Num infer: " << num_infer << "\n";
    std::cout << "Average postprocess time:\t" << sum_postprocess_time / num_postprocess << " ms, Num postprocess: " << num_postprocess << "\n";
    std::cout << "Average Turnaround time:\t" << sum_turnaround / num_turnaround / batch_size << " ms\n";
    std::cout << "Maximum Turnaround time:\t" << max_turnaround / num_turnaround << " ms, Num turnaroud: " << num_turnaround << "\n";
    std::cout << "Turnaround time for each data:\t" << max_turnaround / num_turnaround / batch_size << " ms\n"; 
    std::cout << "Application latency:\t" << application_latency / 1000.0 << " s\n";

    chdir(current_dir);
    
    std::ofstream fout("result.txt", std::ios::app);
    fout << sum_turnaround / num_turnaround / batch_size << " " << max_turnaround / num_turnaround << " " << sum_preprocess_time / num_preproces << " " << sum_postprocess_time / num_postprocess  << " " << application_latency / 1000.0 << std::endl;

    // Write json file
    json_object_object_add(json_result, "categories", json_categories);
    json_object_object_add(json_result, "images", json_images);
    json_object_object_add(json_result, "annotations", json_annotations);

    json_object_to_file(result_path, json_result);

    json_object_put(json_result);

    return true;
}
