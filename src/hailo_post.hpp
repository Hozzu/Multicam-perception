#ifndef _HAILOPOST_HPP_
#define _HAILOPOST_HPP_

#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "engine_interface.hpp"
#include "detection.hpp"

// Define a local bounding box structure for NMS buffer parsing.
// Pragma pack ensures the compiler does not add padding bytes, 
// matching the exact memory layout of the hardware output.
#pragma pack(push, 1)
struct internal_bbox_uint16_t {
    uint16_t y_min;
    uint16_t x_min;
    uint16_t y_max;
    uint16_t x_max;
    uint16_t score;
};
#pragma pack(pop)

/**
 * Direct post-processing for Hailo models without relying on the HailoRT API.
 * Returns true on success, false on failure.
 */
static bool hailo_post(tflite::Interpreter* interpreter, int model_mode, std::vector<DetResult>& results, int cur_batch = 0, float score_threshold = 0.001f, float iou_threshold = 0.7f) {
    if (!interpreter){
        std::cerr << "ERROR: Interpreter is null in hailo_post\n";
        return false;
    }

    // NMS Parameters
    int max_detections = 300;
    float conf_threshold = interpreter->GetHailoScoreThreshold() > 0 ? interpreter->GetHailoScoreThreshold() : score_threshold;

    switch(model_mode) {
        case 1: // SSD MobileNet
        case 3: // EfficientDet Lite
        case 4: // YOLO
        case 5: // YOLOv10
        {
            int nms_out_idx = -1;
            TfLiteTensor* nms_tensor = nullptr;

            // Step 1: Identify the NMS output tensor dynamically
            for(size_t i = 0; i < interpreter->hailo_outputs().size(); ++i) {
                TfLiteTensor* t = interpreter->hailo_output_tensor(i);
                if(t->name && std::string(t->name).find("nms") != std::string::npos) {
                    nms_tensor = t;
                    nms_out_idx = i;
                    break;
                }
            }

            // Fallback: If no tensor is named "nms", pick the first Hailo output
            if (!nms_tensor && !interpreter->hailo_outputs().empty()) {
                nms_out_idx = 0;
                nms_tensor = interpreter->hailo_output_tensor(nms_out_idx);
            }

            if (!nms_tensor) {
                std::cerr << "ERROR: NMS output tensor not found in Interpreter.\n";
                return false;
            }

            int batch_size = nms_tensor->dims->data[0] > 0 ? nms_tensor->dims->data[0] : 1;
            
            // Infer the number of classes based on the model mode.
            // This prevents reading garbage padding bytes at the end of the dynamically packed NMS buffer.
            int num_classes = (model_mode == 1) ? 91 : 80;

            // =========================================================================
            // Branch A: FLOAT32 Output
            // =========================================================================
            if (nms_tensor->type == kTfLiteFloat32) {
                float* raw_data = interpreter->typed_hailo_output_tensor<float>(nms_out_idx, cur_batch);
                if (!raw_data) {
                    std::cerr << "ERROR: Failed to get raw data for NMS output tensor.\n";
                    return false;
                }

                size_t max_floats_per_batch = (nms_tensor->bytes / batch_size) / sizeof(float);
                size_t offset = 0;

                // Loop exactly 'num_classes' times
                for (int class_index = 1; class_index <= num_classes; ++class_index) {
                    if (offset >= max_floats_per_batch) break;

                    float bbox_count_f = raw_data[offset++];
                    int bbox_count = static_cast<int>(bbox_count_f);

                    if (bbox_count > 0) {
                        for (int b = 0; b < bbox_count; b++) {
                            // 5 floats per bbox (ymin, xmin, ymax, xmax, score)
                            if (offset + 5 > max_floats_per_batch) break;

                            float ymin = raw_data[offset++];
                            float xmin = raw_data[offset++];
                            float ymax = raw_data[offset++];
                            float xmax = raw_data[offset++];
                            float score = raw_data[offset++];

                            // Filter by default confidence threshold
                            if (score > conf_threshold) {
                                DetResult res;
                                res.score = score;
                                res.id = class_index - 1; // 1-indexed internally
                                res.xmin = xmin;
                                res.ymin = ymin;
                                res.xmax = xmax;
                                res.ymax = ymax;
                                results.push_back(res);
                            }
                        }
                    }
                }
            }
            // =========================================================================
            // Branch B: UINT16 Output
            // =========================================================================
            else if (nms_tensor->type == kTfLiteUInt16) {
                uint8_t* raw_data = interpreter->typed_hailo_output_tensor<uint8_t>(nms_out_idx, cur_batch);
                if (!raw_data) {
                    std::cerr << "ERROR: Failed to get raw data for NMS output tensor.\n";
                    return false;
                }

                float scale = nms_tensor->params.scale;
                float zero_point = static_cast<float>(nms_tensor->params.zero_point);
                size_t max_bytes_per_batch = nms_tensor->bytes / batch_size;
                size_t offset = 0;

                for (int class_index = 1; class_index <= num_classes; ++class_index) {
                    if (offset + sizeof(uint16_t) > max_bytes_per_batch) break;

                    uint16_t bbox_count = *reinterpret_cast<uint16_t*>(raw_data + offset);
                    offset += sizeof(uint16_t);

                    if (bbox_count > 0) {
                        for (uint16_t b = 0; b < bbox_count; b++) {
                            if (offset + sizeof(internal_bbox_uint16_t) > max_bytes_per_batch) break;

                            internal_bbox_uint16_t* box = reinterpret_cast<internal_bbox_uint16_t*>(raw_data + offset);
                            offset += sizeof(internal_bbox_uint16_t);

                            float score = (box->score - zero_point) * scale;
                            
                            if (score > conf_threshold) {
                                DetResult res;
                                res.score = score;
                                res.id = class_index - 1;
                                res.xmin = (box->x_min - zero_point) * scale;
                                res.ymin = (box->y_min - zero_point) * scale;
                                res.xmax = (box->x_max - zero_point) * scale;
                                res.ymax = (box->y_max - zero_point) * scale;
                                results.push_back(res);
                            }
                        }
                    }
                }
            } 
            else {
                std::cerr << "ERROR: Unsupported tensor type for Hailo NMS output.\n";
                return false;
            }

            // Step 3: Sort results by confidence score descending
            std::sort(results.begin(), results.end(), [](const DetResult& a, const DetResult& b) {
                return a.score > b.score;
            });

            return true; // Success
        }
        case 7: // DETR ResNet
        {
            int num_classes = 92;
            int num_queries = 100;
            int num_coords = 4;

            int output_logit_idx = -1;
            int output_box_idx = -1;

            if (interpreter->hailo_outputs().size() < 2) {
                std::cerr << "ERROR: DETR ResNet requires at least 2 outputs.\n";
                return false;
            }

            TfLiteTensor* output_tensor_0 = interpreter->hailo_output_tensor(0);
            TfLiteIntArray* output_dims_0 = output_tensor_0->dims;

            TfLiteTensor* output_tensor_1 = interpreter->hailo_output_tensor(1);
            TfLiteIntArray* output_dims_1 = output_tensor_1->dims;

            // Determine which output is logits and which is boxes based on shape dimensions
            if (output_dims_0->data[1] == num_queries && output_dims_0->data[2] == num_classes) {
                output_logit_idx = 0;
                if (output_dims_1->data[1] == num_queries && output_dims_1->data[2] == num_coords) {
                    output_box_idx = 1;
                } else { 
                    std::cerr << "ERROR: Invalid output tensor dimensions for DETR ResNet.\n";
                    return false;
                }
            }
            else if (output_dims_0->data[1] == num_queries && output_dims_0->data[2] == num_coords) {
                output_box_idx = 0;
                if (output_dims_1->data[1] == num_queries && output_dims_1->data[2] == num_classes) {
                    output_logit_idx = 1;
                } else { 
                    std::cerr << "ERROR: Invalid output tensor dimensions for DETR ResNet.\n";
                    return false;
                }
            }
            else { 
                std::cerr << "ERROR: Invalid output tensor dimensions for DETR ResNet.\n";
                return false;
            }

            TfLiteTensor* logit_tensor = interpreter->hailo_output_tensor(output_logit_idx);
            TfLiteTensor* box_tensor = interpreter->hailo_output_tensor(output_box_idx);

            results.reserve(num_queries);

            // =========================================================================
            // Branch A: FLOAT32 Output
            // =========================================================================
            if (logit_tensor->type == kTfLiteFloat32 && box_tensor->type == kTfLiteFloat32) {
                float* output_logits = interpreter->typed_hailo_output_tensor<float>(output_logit_idx, cur_batch);
                float* output_boxes = interpreter->typed_hailo_output_tensor<float>(output_box_idx, cur_batch);
                if (!output_logits || !output_boxes) {
                    std::cerr << "ERROR: Failed to get raw data for DETR ResNet output tensors.\n";
                    return false;
                }

                for (int i = 0; i < num_queries; i++) {
                    std::vector<float> scores;
                    scores.reserve(num_classes);
                    for(int j = 0; j < num_classes; ++j) { 
                        scores.push_back(output_logits[i * num_classes + j]);
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

                    auto max_it = std::max_element(scores.begin(), scores.end() - 1); 
                    float max_score = *max_it;
                    int class_id = std::distance(scores.begin(), max_it);

                    float dq_cx = output_boxes[i * 4 + 0];
                    float dq_cy = output_boxes[i * 4 + 1];
                    float dq_w  = output_boxes[i * 4 + 2];
                    float dq_h  = output_boxes[i * 4 + 3];
                    
                    float cx = 1.0f / (1.0f + std::exp(-dq_cx));
                    float cy = 1.0f / (1.0f + std::exp(-dq_cy));
                    float w = 1.0f / (1.0f + std::exp(-dq_w));
                    float h = 1.0f / (1.0f + std::exp(-dq_h));

                    DetResult result;
                    result.score = max_score;
                    result.xmin = (cx - 0.5f * w);
                    result.ymin = (cy - 0.5f * h);
                    result.xmax = (cx + 0.5f * w);
                    result.ymax = (cy + 0.5f * h);
                    result.id = class_id - 1; 

                    results.push_back(result);
                }
            }
            // =========================================================================
            // Branch B: UINT16 Output
            // =========================================================================
            else if (logit_tensor->type == kTfLiteUInt16 && box_tensor->type == kTfLiteUInt16) {
                float logit_scale = logit_tensor->params.scale;
                float logit_zero_point = static_cast<float>(logit_tensor->params.zero_point);

                float box_scale = box_tensor->params.scale;
                float box_zero_point = static_cast<float>(box_tensor->params.zero_point);

                uint16_t* output_logits = interpreter->typed_hailo_output_tensor<uint16_t>(output_logit_idx, cur_batch);
                uint16_t* output_boxes = interpreter->typed_hailo_output_tensor<uint16_t>(output_box_idx, cur_batch);

                if (!output_logits || !output_boxes) {
                    std::cerr << "ERROR: Failed to get raw data for DETR ResNet output tensors.\n";
                    return false;
                }

                for (int i = 0; i < num_queries; i++) {
                    std::vector<float> scores;
                    scores.reserve(num_classes);
                    for(int j = 0; j < num_classes; ++j) { 
                        scores.push_back((output_logits[i * num_classes + j] - logit_zero_point) * logit_scale);
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

                    auto max_it = std::max_element(scores.begin(), scores.end() - 1); 
                    float max_score = *max_it;
                    int class_id = std::distance(scores.begin(), max_it);

                    float dq_cx = (output_boxes[i * 4 + 0] - box_zero_point) * box_scale;
                    float dq_cy = (output_boxes[i * 4 + 1] - box_zero_point) * box_scale;
                    float dq_w  = (output_boxes[i * 4 + 2] - box_zero_point) * box_scale;
                    float dq_h  = (output_boxes[i * 4 + 3] - box_zero_point) * box_scale;
                    
                    float cx = 1.0f / (1.0f + std::exp(-dq_cx));
                    float cy = 1.0f / (1.0f + std::exp(-dq_cy));
                    float w = 1.0f / (1.0f + std::exp(-dq_w));
                    float h = 1.0f / (1.0f + std::exp(-dq_h));

                    DetResult result;
                    result.score = max_score;
                    result.xmin = (cx - 0.5f * w);
                    result.ymin = (cy - 0.5f * h);
                    result.xmax = (cx + 0.5f * w);
                    result.ymax = (cy + 0.5f * h);
                    result.id = class_id - 1; 

                    results.push_back(result);
                }
            } else {
                std::cerr << "ERROR: Unsupported tensor type for DETR output.\n";
                return false;
            }

            std::sort(results.begin(), results.end(), [](const DetResult& a, const DetResult& b) {
                return a.score > b.score;
            });

            return true; 
        }
        default:
            std::cerr << "ERROR: Unsupported model mode for hailo_post: " << model_mode << "\n";
            return false;
    }
}

#endif //_HAILOPOST_HPP_