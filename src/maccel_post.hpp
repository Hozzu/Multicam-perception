#ifndef _MACCELPOST_HPP_
#define _MACCELPOST_HPP_

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

#include <engine_interface.hpp>
#include <opencv2/opencv.hpp>
#include "detection.hpp"


static bool maccel_post(tflite::Interpreter * interpreter, int model_mode, std::vector<DetResult> & results, bool multi = false, int cur_batch = 0, float score_threshold = 0.001f, float iou_threshold = 0.7f){
    if (!interpreter){
        std::cerr << "ERROR: Interpreter is null in maccel_post\n";
        return false;
    }

    // NMS Parameters
    int max_detections = 300;
    float conf_threshold = interpreter->GetMaccelScoreThreshold() > 0 ? interpreter->GetMaccelScoreThreshold() : 0.001f;
    bool multi_label = multi; 
    
    switch(model_mode){
        case 4: // yolo
        {
            // --- 1. Get Input Tensor Info ---
            TfLiteTensor* input_tensor_0 = interpreter->maccel_input_tensor(0);
            TfLiteIntArray* input_dims = input_tensor_0->dims;
            int input_height = input_dims->data[1];
            int input_width = input_dims->data[2];

            // Prepare containers for detections
            std::vector<int> class_ids;
            std::vector<float> scores;
            std::vector<cv::Rect2d> boxes;

            // --- 2. Check Output Topology ---
            int num_outputs = interpreter->maccel_outputs().size();

            // =========================================================================
            // BRANCH A: Single Processed Output (e.g., 1x84x8400 or 1x8400x84)
            // =========================================================================
            if (num_outputs == 1) {
                TfLiteTensor* output_tensor_0 = interpreter->maccel_output_tensor(0);
                TfLiteIntArray* output_dims = output_tensor_0->dims;
                
                // Determine layout: Channels-First (84x8400) or Channels-Last (8400x84)
                int dim_A = output_dims->data[output_dims->size - 2];
                int dim_B = output_dims->data[output_dims->size - 1];

                int num_anchors = 0;
                int num_channels = 0;
                bool is_channel_last = false;

                if (dim_B > dim_A) { // 1x84x8400
                    num_channels = dim_A;
                    num_anchors = dim_B;
                    is_channel_last = false;
                } else { // 1x8400x84
                    num_anchors = dim_A;
                    num_channels = dim_B;
                    is_channel_last = true;
                }

                float * output = interpreter->typed_maccel_output_tensor<float>(0, cur_batch);

                for(int i = 0; i < num_anchors; i++){
                    float x, y, w, h;
                    
                    // Extract coordinates based on layout
                    if (is_channel_last) {
                        int offset = i * num_channels;
                        x = output[offset + 0]; y = output[offset + 1];
                        w = output[offset + 2]; h = output[offset + 3];
                    } else {
                        x = output[0 * num_anchors + i]; y = output[1 * num_anchors + i];
                        w = output[2 * num_anchors + i]; h = output[3 * num_anchors + i];
                    }

                    float xmin = (x - 0.5f * w) * input_width;
                    float ymin = (y - 0.5f * h) * input_height;
                    float width = w * input_width;
                    float height = h * input_height;

                    float max_score = 0;
                    int max_class_id = 0;

                    // Find best class score
                    for(int j = 4; j < num_channels; j++){
                        float score;
                        if (is_channel_last) score = output[i * num_channels + j];
                        else                 score = output[j * num_anchors + i];

                        int class_id = j - 4;
                        if(score > max_score){
                            max_score = score;
                            max_class_id = class_id;
                        }
                        
                        if(multi_label && score > conf_threshold){
                            scores.push_back(score);
                            class_ids.push_back(class_id);
                            boxes.push_back(cv::Rect2d(xmin, ymin, width, height));
                        }
                    }

                    if(!multi_label && max_score > conf_threshold){
                        scores.push_back(max_score);
                        class_ids.push_back(max_class_id);
                        boxes.push_back(cv::Rect2d(xmin, ymin, width, height));
                    }
                }
            }
            // =========================================================================
            // BRANCH B: Raw Feature Maps (3 or 6 Outputs) - Requires DFL Decoding
            // =========================================================================
            else if (num_outputs == 3 || num_outputs == 6) {
                int reg_max = 16; // DFL channels per coordinate (64 / 4)
                int num_scales = num_outputs == 6 ? 3 : num_outputs; 

                // Iterate over feature map scales
                for (int s = 0; s < num_scales; s++) {
                    float* reg_data = nullptr;
                    float* cls_data = nullptr;
                    int grid_h, grid_w;
                    int reg_channels = 0; 
                    int cls_channels = 0;

                    // --- Configure pointers and shapes based on output count ---
                    if (num_outputs == 6) {
                        // Pair: (Reg, Cls), (Reg, Cls), ...
                        int reg_idx = s * 2;
                        int cls_idx = s * 2 + 1;
                        
                        TfLiteTensor* t_reg = interpreter->maccel_output_tensor(reg_idx);
                        TfLiteTensor* t_cls = interpreter->maccel_output_tensor(cls_idx);

                        grid_h = t_reg->dims->data[1];
                        grid_w = t_reg->dims->data[2];
                        reg_channels = t_reg->dims->data[3]; // 64
                        cls_channels = t_cls->dims->data[3]; // 80

                        reg_data = interpreter->typed_maccel_output_tensor<float>(reg_idx, cur_batch);
                        cls_data = interpreter->typed_maccel_output_tensor<float>(cls_idx, cur_batch);
                    } 
                    else if (num_outputs == 3) {
                        // Combined: [Reg(64) + Cls(80)] = 144
                        int out_idx = s;
                        TfLiteTensor* t_out = interpreter->maccel_output_tensor(out_idx);

                        grid_h = t_out->dims->data[1];
                        grid_w = t_out->dims->data[2];
                        int total_channels = t_out->dims->data[3]; // 144
                        
                        reg_channels = 64; // 4 * 16
                        cls_channels = total_channels - reg_channels; // 80

                        float* raw_data = interpreter->typed_maccel_output_tensor<float>(out_idx, cur_batch);
                        reg_data = raw_data; // Points to start (Reg)
                        cls_data = raw_data + reg_channels; // Offset for Cls, handled in loop
                    }

                    float stride_x = (float)input_width / grid_w;
                    float stride_y = (float)input_height / grid_h;

                    // Iterate over Grid
                    for (int h = 0; h < grid_h; h++) {
                        for (int w = 0; w < grid_w; w++) {
                            
                            // 1. Process Classification
                            // Calculate pointer offsets
                            // For 6 outputs: Separate buffers.
                            // For 3 outputs: Same buffer, but stride logic needs care.
                            // Assuming memory layout is [Batch, H, W, Channels] (NHWC)
                            
                            int pixel_idx = (h * grid_w + w);
                            
                            float max_cls_score = -1.0f;
                            int max_cls_id = -1;

                            // Pointer to current pixel's class data
                            float* current_cls_ptr = nullptr;
                            if(num_outputs == 6) current_cls_ptr = cls_data + pixel_idx * cls_channels;
                            else                 current_cls_ptr = reg_data + pixel_idx * (reg_channels + cls_channels) + reg_channels;

                            for (int c = 0; c < cls_channels; c++) {
                                float raw_score = current_cls_ptr[c];
                                // Apply Sigmoid to convert logit to probability
                                float score = 1.0f / (1.0f + std::exp(-raw_score)); 

                                if (score > max_cls_score) {
                                    max_cls_score = score;
                                    max_cls_id = c;
                                }
                            }

                            if (max_cls_score < conf_threshold) continue;

                            // 2. Process Regression (DFL)
                            // Pointer to current pixel's reg data
                            float* current_reg_ptr = nullptr;
                            if(num_outputs == 6) current_reg_ptr = reg_data + pixel_idx * reg_channels;
                            else                 current_reg_ptr = reg_data + pixel_idx * (reg_channels + cls_channels);

                            float pred_dist[4]; // l, t, r, b
                            
                            for (int k = 0; k < 4; k++) {
                                float exp_sum = 0.0f;
                                float weighted_sum = 0.0f;
                                float max_val = -1e9;
                                
                                // Pointer to the 16 bins for this coordinate
                                float* bins = current_reg_ptr + k * reg_max;

                                // Find max for numerical stability
                                for(int r=0; r<reg_max; r++) if(bins[r] > max_val) max_val = bins[r];

                                // Softmax + Integrate
                                for (int r = 0; r < reg_max; r++) {
                                    float e = std::exp(bins[r] - max_val);
                                    exp_sum += e;
                                    weighted_sum += e * r;
                                }
                                pred_dist[k] = weighted_sum / exp_sum;
                            }

                            // 3. Decode Box (Dist to xyxy)
                            // Anchor center
                            float cx = w + 0.5f;
                            float cy = h + 0.5f;

                            float x1 = (cx - pred_dist[0]) * stride_x;
                            float y1 = (cy - pred_dist[1]) * stride_y;
                            float x2 = (cx + pred_dist[2]) * stride_x;
                            float y2 = (cy + pred_dist[3]) * stride_y;

                            boxes.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                            scores.push_back(max_cls_score);
                            class_ids.push_back(max_cls_id);
                        }
                    }
                }
            }

            // =========================================================================
            // 3. Common NMS (Shared for all branches)
            // =========================================================================
            std::vector<int> nms_result;
            if(multi_label){
                std::vector<cv::Rect2d> _boxes = boxes;
                for(int i = 0; i < _boxes.size(); i++){
                    cv::Point2d offset(class_ids[i] * input_width, class_ids[i] * input_height);
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
                DetResult result;
                result.score = scores[idx];
                result.xmin = boxes[idx].x / input_width;
                result.ymin = boxes[idx].y / input_height;
                result.xmax = (boxes[idx].x + boxes[idx].width) / input_width;
                result.ymax = (boxes[idx].y + boxes[idx].height) / input_height;
                result.id = class_ids[idx];
                results.push_back(result);
            }

            break;
        }
        case 5: // yolov10
        {
            // --- 1. Get Input Tensor Info ---
            TfLiteTensor* input_tensor_0 = interpreter->maccel_input_tensor(0);
            TfLiteIntArray* input_dims = input_tensor_0->dims;
            int input_height = input_dims->data[1];
            int input_width = input_dims->data[2];
            
            int num_outputs = interpreter->maccel_outputs().size();

            // Case A: End-to-End Output (1x300x6)
            if (num_outputs == 1) {
                TfLiteTensor* output_tensor = interpreter->maccel_output_tensor(0);
                TfLiteIntArray* dims = output_tensor->dims;
                int rows = dims->data[1]; // 300
                int cols = dims->data[2]; // 6

                float* data = interpreter->typed_maccel_output_tensor<float>(0, cur_batch);
                float (*data_arr)[cols] = (float(*)[cols])data;

                results.reserve(rows);
                for(int i = 0; i < rows; i++){
                    float score = data_arr[i][4];
                    if(score < conf_threshold) continue;

                    DetResult result;
                    result.xmin = data_arr[i][0];
                    result.ymin = data_arr[i][1];
                    result.xmax = data_arr[i][2];
                    result.ymax = data_arr[i][3];
                    result.score = score;
                    result.id = (int)data_arr[i][5];
                    results.push_back(result);
                }
            }
            // Case B: Raw Feature Maps (3 Scales x 2 Branches = 6 Outputs)
            else if (num_outputs == 6) {
                // Parameters
                const int strides[3] = {32, 16, 8}; // Output 0,1 -> 20x20 (Stride 32), Output 4,5 -> 80x80 (Stride 8)
                const int reg_max = 16; // 64 channels / 4 coords

                std::vector<int> class_ids;
                std::vector<float> scores;
                std::vector<cv::Rect2d> boxes;

                // Iterate over 3 scales
                // Output indices assumed: 
                // 0: 20x20x64(Reg), 1: 20x20x80(Cls) -> Stride 32
                // 2: 40x40x64(Reg), 3: 40x40x80(Cls) -> Stride 16
                // 4: 80x80x64(Reg), 5: 80x80x80(Cls) -> Stride 8
                for (int s = 0; s < 3; s++) {
                    int reg_idx = s * 2;
                    int cls_idx = s * 2 + 1;
                    int stride = strides[s];

                    TfLiteTensor* reg_tensor = interpreter->maccel_output_tensor(reg_idx);
                    TfLiteTensor* cls_tensor = interpreter->maccel_output_tensor(cls_idx);
                    
                    // Assume format 1xHxWxChannel (Channel Last)
                    int height = reg_tensor->dims->data[1];
                    int width = reg_tensor->dims->data[2];
                    int reg_channels = reg_tensor->dims->data[3]; // 64
                    int cls_channels = cls_tensor->dims->data[3]; // 80

                    float* reg_data = interpreter->typed_maccel_output_tensor<float>(reg_idx, cur_batch);
                    float* cls_data = interpreter->typed_maccel_output_tensor<float>(cls_idx, cur_batch);

                    for (int h = 0; h < height; h++) {
                        for (int w = 0; w < width; w++) {
                            // 1. Process Classification (Find max class score)
                            // Offset for current pixel in Cls tensor
                            int cls_offset = (h * width + w) * cls_channels;
                            
                            float max_cls_score = -1.0f;
                            int max_cls_id = -1;

                            for (int c = 0; c < cls_channels; c++) {
                                float raw_score = cls_data[cls_offset + c];
                                // Apply Sigmoid: 1 / (1 + exp(-x))
                                float score = 1.0f / (1.0f + std::exp(-raw_score)); 

                                if (score > max_cls_score) {
                                    max_cls_score = score;
                                    max_cls_id = c;
                                }
                            }

                            if (max_cls_score < conf_threshold) continue;

                            // 2. Process Regression (DFL Decoding)
                            // Offset for current pixel in Reg tensor
                            int reg_offset = (h * width + w) * reg_channels;
                            float dists[4]; // left, top, right, bottom

                            // Decode 4 coordinates
                            for (int k = 0; k < 4; k++) {
                                // Softmax over 16 bins for this coordinate
                                float exp_sum = 0.0f;
                                std::vector<float> exps(reg_max);
                                float max_val = -1e9; // for numerical stability

                                // Find max for subtraction
                                for(int r=0; r<reg_max; r++) {
                                    float val = reg_data[reg_offset + k * reg_max + r];
                                    if(val > max_val) max_val = val;
                                }

                                for (int r = 0; r < reg_max; r++) {
                                    exps[r] = std::exp(reg_data[reg_offset + k * reg_max + r] - max_val);
                                    exp_sum += exps[r];
                                }

                                // Weighted sum (Integrate)
                                float decoded_val = 0.0f;
                                for (int r = 0; r < reg_max; r++) {
                                    float prob = exps[r] / exp_sum;
                                    decoded_val += prob * r;
                                }
                                dists[k] = decoded_val;
                            }

                            // 3. Convert Distances to BBox (xyxy)
                            // dists are: left, top, right, bottom (relative to anchor center)
                            float cx = w + 0.5f;
                            float cy = h + 0.5f;

                            float x1 = (cx - dists[0]) * stride / input_width;
                            float y1 = (cy - dists[1]) * stride / input_height;
                            float x2 = (cx + dists[2]) * stride / input_width;
                            float y2 = (cy + dists[3]) * stride / input_height;

                            boxes.push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                            scores.push_back(max_cls_score);
                            class_ids.push_back(max_cls_id);
                        }
                    }
                }

                // Apply NMS
                std::vector<int> nms_result;
                cv::dnn::NMSBoxes(boxes, scores, conf_threshold, iou_threshold, nms_result, 1.0, max_detections);

                results.reserve(nms_result.size());
                for (int idx : nms_result) {
                    //std::cout << "Score: " << scores[idx] << ", Box: [" << boxes[idx].x << ", " << boxes[idx].y << ", " << boxes[idx].x + boxes[idx].width << ", " << boxes[idx].y + boxes[idx].height << "], Class ID: " << class_ids[idx] << std::endl;
                    
                    DetResult result;
                    result.score = scores[idx];
                    result.xmin = boxes[idx].x;
                    result.ymin = boxes[idx].y;
                    result.xmax = boxes[idx].x + boxes[idx].width;
                    result.ymax = boxes[idx].y + boxes[idx].height;
                    result.id = class_ids[idx];
                    results.push_back(result);
                }
            }

            break;
        }
        case 7: //detr resnet
        {
            int num_classes = 92;
            int num_queries = 100;
            int num_coords = 4;

            int output_logit_idx;
            int output_box_idx;

            TfLiteTensor* output_tensor_0 = interpreter->maccel_output_tensor(0);
            TfLiteIntArray* output_dims_0 = output_tensor_0->dims;

            TfLiteTensor* output_tensor_1 = interpreter->maccel_output_tensor(1);
            TfLiteIntArray* output_dims_1 = output_tensor_1->dims;

            if(output_dims_0->data[1] == num_queries && output_dims_0->data[2] == num_classes){
                output_logit_idx = 0;
                if(output_dims_1->data[1] == num_queries && output_dims_1->data[2] == num_coords){
                    output_box_idx = 1;
                }
                else{
                    std::cerr << "ERROR: cannot recognize detr resnet output index\n";
                    return false;
                }
            }
            else if(output_dims_0->data[1] == num_queries && output_dims_0->data[2] == num_coords){
                output_box_idx = 0;
                if(output_dims_1->data[1] == num_queries && output_dims_1->data[2] == num_classes){
                    output_logit_idx = 1;
                }
                else{
                    std::cerr << "ERROR: cannot recognize detr resnet output index\n";
                    return false;
                }
            }
            else{
                std::cerr << "ERROR: cannot recognize detr resnet output index\n";
                return false;
            }

            float * output_logits = interpreter->typed_maccel_output_tensor<float>(output_logit_idx, cur_batch);
            float * output_boxes = interpreter->typed_maccel_output_tensor<float>(output_box_idx, cur_batch);

            results.reserve(num_queries);

            for (int i = 0; i < num_queries; i++){
                std::vector<float> scores;
                scores.reserve(num_classes);
                for(int j = 0; j < num_classes; ++j) { 
                    scores.push_back(output_logits[i * num_classes + j]);
                }
                
                // Apply Softmax on logits to get probabilities
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

                float cx = output_boxes[i * 4 + 0];
                float cy = output_boxes[i * 4 + 1];
                float w  = output_boxes[i * 4 + 2];
                float h  = output_boxes[i * 4 + 3];

                // Apply Sigmoid to box coordinates
                cx = 1.0f / (1.0f + std::exp(-cx));
                cy = 1.0f / (1.0f + std::exp(-cy));
                w = 1.0f / (1.0f + std::exp(-w));
                h = 1.0f / (1.0f + std::exp(-h));

                int id = class_id;
                float score = max_score;
                float xmin = (cx - 0.5f * w) > 0 ? (cx - 0.5f * w) : 0;
                float ymin = (cy - 0.5f * h) > 0 ? (cy - 0.5f * h) : 0;
                float xmax = (cx + 0.5f * w) < 1 ? (cx + 0.5f * w) : 1;
                float ymax = (cy + 0.5f * h) < 1 ? (cy + 0.5f * h) : 1;

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

            return true;

            break;
        }
        default:
            std::cerr << "ERROR: Unsupported model mode for maccel_post " << model_mode << "\n"; 
            return false;
    }

    return true;
}

#endif // _MACCELPOST_HPP_