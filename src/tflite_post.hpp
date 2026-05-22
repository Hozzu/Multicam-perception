#ifndef _TFLITEPOST_HPP_
#define _TFLITEPOST_HPP_

#include <vector>

#include <engine_interface.hpp>

#include <opencv2/opencv.hpp>

#include "detection.hpp"


static bool tflite_post(tflite::Interpreter * interpreter, int model_mode, std::vector<DetResult> & results, bool multi = false, int cur_batch = 0, float score_threshold = 0.001f, float iou_threshold = 0.7f){
    if (!interpreter){
        std::cerr << "ERROR: Interpreter is null in tflite_post\n";
        return false;
    }

    // NMS Parameters
    int max_detections = 300;
    float conf_threshold = score_threshold;
    bool multi_label = multi;
    
    switch(model_mode){
        case 1: //ssd_mobilenet
        {
            int output_box_idx = 0;
            int output_class_idx = 1;
            int output_score_idx = 2;
            int output_num_idx = 3;

            float * output_locations = interpreter->typed_tflite_output_tensor<float>(output_box_idx, cur_batch);
            float * output_classes = interpreter->typed_tflite_output_tensor<float>(output_class_idx, cur_batch);
            float * output_scores = interpreter->typed_tflite_output_tensor<float>(output_score_idx, cur_batch);
            int output_nums = (int) *(interpreter->typed_tflite_output_tensor<float>(output_num_idx, cur_batch));

            results.reserve(output_nums);

            for (int i = 0; i < output_nums; i++){
                float score =  output_scores[i];

                float ymin = output_locations[i * 4];
                float xmin = output_locations[i * 4 + 1];
                float ymax = output_locations[i * 4 + 2];
                float xmax = output_locations[i * 4 + 3];

                int id =  (int)(output_classes[i]);

                //std::cout << i <<  ": output_classes: " << id << ", output_scores: " << score << ", output_locations: [" << xmin << "," << ymin << "," << xmax << ","<< ymax << "]\n";

                DetResult result;
                result.score = score;
                result.xmin = xmin;
                result.ymin = ymin;
                result.xmax = xmax;
                result.ymax = ymax;
                result.id =  id;

                results.push_back(result);
            }

            break;
        }
        case 2: //efficientdet
        case 3: //efficientdet lite
        {
            std::vector<std::pair<std::string, int>> tflite_output_tensors;

            for (int i = 0; i < interpreter->tflite_outputs().size(); i++) {
                const TfLiteTensor* tensor = interpreter->tflite_output_tensor(i);
                
                std::string tensor_name = (tensor != nullptr && tensor->name != nullptr) ? std::string(tensor->name) : "";
                tflite_output_tensors.push_back({tensor_name, i});
            }

            // Sort alphabetically by tensor name
            std::sort(tflite_output_tensors.begin(), tflite_output_tensors.end());

            int output_num_local_idx = tflite_output_tensors[0].second;
            int output_score_local_idx = tflite_output_tensors[1].second;
            int output_class_local_idx = tflite_output_tensors[2].second;
            int output_box_local_idx = tflite_output_tensors[3].second;

            float * output_locations = interpreter->typed_tflite_output_tensor<float>(output_box_local_idx, cur_batch);
            float * output_classes = interpreter->typed_tflite_output_tensor<float>(output_class_local_idx, cur_batch);
            float * output_scores = interpreter->typed_tflite_output_tensor<float>(output_score_local_idx, cur_batch);
            int output_nums = (int) *(interpreter->typed_tflite_output_tensor<float>(output_num_local_idx, cur_batch));

            results.reserve(output_nums);

            for(int i = 0; i < output_nums; i++){     
                float score =  output_scores[i];

                float ymin = output_locations[i * 4];
                float xmin = output_locations[i * 4 + 1];
                float ymax = output_locations[i * 4 + 2];
                float xmax = output_locations[i * 4 + 3];

                int id =  (int)(output_classes[i]);

                //std::cout << i <<  ": output_classes: " << id << ", output_scores: " << score << ", output_locations: [" << xmin << "," << ymin << "," << xmax << ","<< ymax << "]\n";

                DetResult result;
                result.score = score;
                result.xmin = xmin;
                result.ymin = ymin;
                result.xmax = xmax;
                result.ymax = ymax;
                result.id =  id;

                results.push_back(result);
            }

            break;
        }
        case 4: //yolo
        {
            // Get the output tensor size info
            TfLiteTensor* output_tensor_0 = interpreter->tflite_output_tensor(0);
            TfLiteIntArray* output_dims = output_tensor_0->dims;
            int output_height = output_dims->data[1];
            int output_width = output_dims->data[2];

            // Parse output and apply nms
            float * output = interpreter->typed_tflite_output_tensor<float>(0, cur_batch);
            float (*output_arr)[output_width] = (float(*)[output_width])output;

            std::vector<int> class_ids;
            std::vector<float> scores;
            std::vector<cv::Rect2d> boxes;

            for(int i = 0; i < output_width; i++){
                float x = output_arr[0][i];
                float y = output_arr[1][i];
                float w = output_arr[2][i];
                float h = output_arr[3][i];

                float xmin = (x - 0.5 * w);
                float ymin = (y - 0.5 * h);
                float width = w;
                float height = h;

                float max_score = 0;
                int max_class_id = 0;

                for(int j = 4; j < output_height; j++){
                    float score = output_arr[j][i];

                    int class_id = j - 4;

                    if(score > max_score){
                        max_score = score;
                        max_class_id = class_id;
                    }

                    if(multi_label){
                        if(score > conf_threshold){
                            scores.push_back(score);
                            class_ids.push_back(class_id);
                            boxes.push_back(cv::Rect2d(xmin, ymin, width, height));
                        }
                    }
                }

                if(!multi_label){
                    if(max_score > conf_threshold){
                        scores.push_back(max_score);
                        class_ids.push_back(max_class_id);
                        boxes.push_back(cv::Rect2d(xmin, ymin, width, height));
                    }
                }
            }
            
            std::vector<int> nms_result;
            if(multi_label){
                // Batched nms trick since batched nms function is only available opencv > 4.7.0
                std::vector<cv::Rect2d> _boxes = boxes;

                for(int i = 0; i < _boxes.size(); i++){
                    cv::Point2d offset(class_ids[i], class_ids[i]);
                    _boxes[i] = _boxes[i] + offset;
                }

                cv::dnn::NMSBoxes(_boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);
            }
            else{
                cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);
            }

            results.reserve(nms_result.size());

            for(int i = 0; i < nms_result.size(); i++){
                int idx = nms_result[i];

                float score = scores[idx];

                float xmin = boxes[idx].x;
                float ymin = boxes[idx].y;
                float xmax = (boxes[idx].x + boxes[idx].width);
                float ymax = (boxes[idx].y + boxes[idx].height);

                int id =  class_ids[idx];

                //std::cout << i <<  ": output_classes: " << id << ", output_scores: " << score << ", output_locations: [" << xmin << "," << ymin << "," << xmax << ","<< ymax << "]\n";

                DetResult result;
                result.score = score;
                result.xmin = xmin;
                result.ymin = ymin;
                result.xmax = xmax;
                result.ymax = ymax;
                result.id =  id;

                results.push_back(result);
            }

            break;
        }
        case 5: //yolov10
        {
            // Get the input tensor size info
            TfLiteTensor* input_tensor_0 = interpreter->tflite_input_tensor(0);
            TfLiteIntArray* input_dims = input_tensor_0->dims;
            int input_height = input_dims->data[1];
            int input_width = input_dims->data[2];
            int input_channel = input_dims->data[3];

            // Get the output tensor size info
            TfLiteTensor* output_tensor_0 = interpreter->tflite_output_tensor(0);
            TfLiteIntArray* output_dims = output_tensor_0->dims;
            int output_height = output_dims->data[1];
            int output_width = output_dims->data[2];

            // Parse output and apply nms
            float * output = interpreter->typed_tflite_output_tensor<float>(0, cur_batch);
            float (*output_arr)[output_width] = (float(*)[output_width])output;

            std::vector<int> class_ids;
            std::vector<float> scores;
            std::vector<cv::Rect2d> boxes;

            for(int i = 0; i < output_width; i++){
                float score = output_arr[i][4];
                if(score < conf_threshold)
                    continue;

                float xmin = output_arr[i][0] / input_height;
                float ymin = output_arr[i][1] / input_width;
                float xmax = output_arr[i][2] / input_height;
                float ymax = output_arr[i][3] / input_width;

                float width = xmax - xmin;
                float height = ymax - ymin;

                int class_id = output_arr[i][5];

                scores.push_back(score);
                class_ids.push_back(class_id);
                boxes.push_back(cv::Rect2d(xmin, ymin, width, height));
            }

            std::vector<int> nms_result;
            cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);

            results.reserve(nms_result.size());
            for (int idx : nms_result) {
                //std::cout << "Score: " << scores[idx] << ", Box: [" << boxes[idx].x << ", " << boxes[idx].y << ", " << boxes[idx].x + boxes[idx].width << ", " << boxes[idx].y + boxes[idx].height << "], Class ID: " << class_ids[idx] << std::endl;
                
                DetResult result;
                result.score = scores[idx];
                result.xmin = boxes[idx].x;
                result.ymin = boxes[idx].y;
                result.xmax = (boxes[idx].x + boxes[idx].width);
                result.ymax = (boxes[idx].y + boxes[idx].height);
                result.id = class_ids[idx];
                results.push_back(result);
            }

            break;
        }
        case 7: //detr resnet
        {
            int num_classes = 92, num_queries = 100;
            int num_coords = 4;
            int output_box_idx = -1, output_logit_idx = -1;

            // Flags to determine memory layout:
            // false: [1, Channels, Queries] (NCQ) - Default DETR output
            // true:  [1, Queries, Channels] (NQC) - Channels Last
            bool logits_are_channels_last = false;
            bool boxes_are_channels_last = false;

            // Iterate through the two output tensors to identify which is which based on dimensions
            for (int k = 0; k < 2; ++k) {
                int tensor_index = k;
                TfLiteTensor* tensor = interpreter->tflite_output_tensor(tensor_index);
                TfLiteIntArray* dims = tensor->dims;

                // Check dimensions to identify Logits (Classes) tensor
                // Case 1: [1, 92, 100] -> Channels First
                if (dims->data[1] == num_classes && dims->data[2] == num_queries) {
                    output_logit_idx = tensor_index;
                    logits_are_channels_last = false;
                }
                // Case 2: [1, 100, 92] -> Channels Last
                else if (dims->data[1] == num_queries && dims->data[2] == num_classes) {
                    output_logit_idx = tensor_index;
                    logits_are_channels_last = true;
                }
                
                // Check dimensions to identify Boxes (Coordinates) tensor
                // Case 1: [1, 4, 100] -> Channels First
                else if (dims->data[1] == num_coords && dims->data[2] == num_queries) {
                    output_box_idx = tensor_index;
                    boxes_are_channels_last = false;
                }
                // Case 2: [1, 100, 4] -> Channels Last
                else if (dims->data[1] == num_queries && dims->data[2] == num_coords) {
                    output_box_idx = tensor_index;
                    boxes_are_channels_last = true;
                }
            }

            // Validate that we found both tensors
            if (output_logit_idx == -1 || output_box_idx == -1) {
                std::cerr << "ERROR: cannot recognize detr resnet output index or shape\n";
                return false;
            }

            float * output_logits = interpreter->typed_tflite_output_tensor<float>(output_logit_idx, cur_batch);
            float * output_boxes = interpreter->typed_tflite_output_tensor<float>(output_box_idx, cur_batch);

            results.reserve(num_queries);

            for (int i = 0; i < num_queries; i++){
                // 1. Process Logits (Apply Softmax)
                std::vector<float> scores;
                scores.reserve(num_classes);

                if (logits_are_channels_last) {
                    // Shape: [Batch, Queries, Classes] -> [1, 100, 92]
                    // Memory is continuous for a single query: [class0, class1, ..., class91]
                    float* current_query_logits = output_logits + (i * num_classes);
                    for(int j = 0; j < num_classes; ++j) {
                        scores.push_back(current_query_logits[j]);
                    }
                } else {
                    // Shape: [Batch, Classes, Queries] -> [1, 92, 100]
                    // Memory jumps by num_queries to get the next class for the same query
                    for(int j = 0; j < num_classes; ++j) {
                        scores.push_back(output_logits[j * num_queries + i]);
                    }
                }

                float max_val = *std::max_element(scores.begin(), scores.end());
                float sum = 0.0f;
                for (size_t j = 0; j < scores.size(); ++j) {
                    scores[j] = std::exp(scores[j] - max_val);
                    sum += scores[j];
                }
                for (size_t j = 0; j < scores.size(); ++j) {
                    scores[j] /= sum;
                }

                // Find the class with the highest probability
                auto max_it = std::max_element(scores.begin(), scores.end() - 1); // Exclude the last class (no-object class)
                float max_score = *max_it;
                int class_id = std::distance(scores.begin(), max_it);

                // 2. Process Boxes
                float cx, cy, w, h;

                if (boxes_are_channels_last) {
                    // Shape: [Batch, Queries, Coords] -> [1, 100, 4]
                    // Layout: [q0_c0, q0_c1, q0_c2, q0_c3, q1_c0...]
                    int offset = i * num_coords;
                    cx = output_boxes[offset + 0];
                    cy = output_boxes[offset + 1];
                    w  = output_boxes[offset + 2];
                    h  = output_boxes[offset + 3];
                } else {
                    // Shape: [Batch, Coords, Queries] -> [1, 4, 100]
                    // Layout: [c0_q0, c0_q1... c1_q0...]
                    cx = output_boxes[i + 0 * num_queries];
                    cy = output_boxes[i + 1 * num_queries];
                    w  = output_boxes[i + 2 * num_queries];
                    h  = output_boxes[i + 3 * num_queries];
                }

                int id = class_id; 
                float score = max_score;
                float xmin = (cx - 0.5f * w);
                float ymin = (cy - 0.5f * h);
                float xmax = (cx + 0.5f * w);
                float ymax = (cy + 0.5f * h);

                DetResult result;
                result.score = score;
                result.xmin = xmin;
                result.ymin = ymin;
                result.xmax = xmax;
                result.ymax = ymax;
                result.id =  id - 1; // Detr class IDs are offset by 1

                results.push_back(result);
            }

            std::sort(results.begin(), results.end(), [](const DetResult& a, const DetResult& b) {
                return a.score > b.score;
            });

            //for (const auto& res : results)
            //    std::cout << "ID: " << res.id << ", Score: " << res.score << ", Box: [" << res.xmin << ", " << res.ymin << ", " << res.xmax << ", " << res.ymax << "]\n";

            break;
        }
        case 8: // rt-detr
        {
            // Get the output tensor.
            TfLiteTensor* output_tensor = interpreter->tflite_output_tensor(0);
            TfLiteIntArray* output_dims = output_tensor->dims;

            // Expected shape: [batch, num_queries, 4 + num_classes] e.g., [1, 300, 84]
            int num_queries = output_dims->data[1];
            int num_details = output_dims->data[2];
            int num_classes = num_details - 4;

            float* output_data = interpreter->typed_tflite_output_tensor<float>(0, cur_batch);

            std::vector<int> class_ids;
            std::vector<float> scores;
            std::vector<cv::Rect2d> boxes;

            for (int i = 0; i < num_queries; ++i) {
                float* current_detection = output_data + i * num_details;

                // Extract box [cx, cy, w, h] and scores
                float cx = current_detection[0];
                float cy = current_detection[1];
                float w = current_detection[2];
                float h = current_detection[3];
                float* class_scores = current_detection + 4;

                // Find the class with the highest score
                float max_score = 0.0f;
                int class_id = -1;
                for (int j = 0; j < num_classes; ++j) {
                    if (class_scores[j] > max_score) {
                        max_score = class_scores[j];
                        class_id = j;
                    }
                }

                if (max_score > conf_threshold) {
                    scores.push_back(max_score);
                    class_ids.push_back(class_id);

                    // Convert [cx, cy, w, h] to [x, y, w, h] for cv::Rect2d
                    float x = cx - w / 2.0f;
                    float y = cy - h / 2.0f;
                    boxes.push_back(cv::Rect2d(x, y, w, h));
                }
            }

            // Apply Non-Maximum Suppression
            std::vector<int> nms_result;
            cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);
            
            results.reserve(nms_result.size());
            for (int idx : nms_result) {
                //std::cout << "Score: " << scores[idx] << ", Box: [" << boxes[idx].x << ", " << boxes[idx].y << ", " << boxes[idx].x + boxes[idx].width << ", " << boxes[idx].y + boxes[idx].height << "], Class ID: " << class_ids[idx] << std::endl;

                DetResult result;
                result.score = scores[idx];
                result.id = class_ids[idx];
                result.xmin = boxes[idx].x;
                result.ymin = boxes[idx].y;
                result.xmax = boxes[idx].x + boxes[idx].width;
                result.ymax = boxes[idx].y + boxes[idx].height;
                results.push_back(result);
            }

            break;
        }
        default:
            std::cerr << "ERROR: Unsupported model mode for tflite_post " << model_mode << "\n";
            return false;
    }

    return true;
}

#endif // _TFLITEPOST_HPP_