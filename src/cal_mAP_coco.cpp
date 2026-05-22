#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <json-c/json.h>

struct BBox {
    double x, y, w, h;
};

struct GTAnnotation {
    int id;
    int image_id;
    int category_id;
    BBox bbox;
};

struct DetAnnotation {
    int image_id;
    int category_id;
    BBox bbox;
    double score;
};

double calculate_iou(const BBox& a, const BBox& b) {
    double x1 = std::max(a.x, b.x);
    double y1 = std::max(a.y, b.y);
    double x2 = std::min(a.x + a.w, b.x + b.w);
    double y2 = std::min(a.y + a.h, b.y + b.h);

    double inter_w = std::max(0.0, x2 - x1);
    double inter_h = std::max(0.0, y2 - y1);
    double inter_area = inter_w * inter_h;

    double area_a = a.w * a.h;
    double area_b = b.w * b.h;

    return inter_area / (area_a + area_b - inter_area);
}

double compute_ap(const std::vector<int>& tp, const std::vector<int>& fp, int total_gt) {
    if (total_gt == 0) return 0.0;
    if (tp.empty()) return 0.0;

    std::vector<double> prec(tp.size());
    std::vector<double> rec(tp.size());
    int acc_tp = 0;
    int acc_fp = 0;

    for (size_t i = 0; i < tp.size(); ++i) {
        acc_tp += tp[i];
        acc_fp += fp[i];
        prec[i] = static_cast<double>(acc_tp) / (acc_tp + acc_fp);
        rec[i] = static_cast<double>(acc_tp) / total_gt;
    }

    // 101-point interpolation (COCO standard)
    double ap = 0.0;
    for (int i = 0; i <= 100; ++i) {
        double t = i / 100.0;
        double p_max = 0.0;
        for (size_t j = 0; j < prec.size(); ++j) {
            if (rec[j] >= t) {
                p_max = std::max(p_max, prec[j]);
            }
        }
        ap += p_max;
    }
    return ap / 101.0;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <gt_json> <det_json>\n";
        return 1;
    }

    const char* gt_file = argv[1];
    const char* det_file = argv[2];

    json_object* gt_root = json_object_from_file(gt_file);
    if (!gt_root) {
        std::cerr << "Error: Failed to parse GT file.\n";
        return 1;
    }

    // --- 1. Parse Categories from GT ---
    // This defines the official classes to be evaluated.
    std::map<int, std::string> categories;
    json_object* j_categories = nullptr;
    if (json_object_object_get_ex(gt_root, "categories", &j_categories) && json_object_get_type(j_categories) == json_type_array) {
        int num_cats = json_object_array_length(j_categories);
        for (int i = 0; i < num_cats; ++i) {
            json_object* cat = json_object_array_get_idx(j_categories, i);
            json_object *j_id, *j_name;
            if (json_object_object_get_ex(cat, "id", &j_id) && json_object_object_get_ex(cat, "name", &j_name)) {
                categories[json_object_get_int(j_id)] = json_object_get_string(j_name);
            }
        }
    }

    if (categories.empty()) {
        std::cerr << "Error: No categories found in Ground Truth JSON.\n";
        json_object_put(gt_root);
        return 1;
    }

    // --- 2. Parse GT Images (Filename to ID mapping) ---
    std::map<std::string, int> gt_filename_to_id;
    json_object* gt_images = nullptr;
    if (json_object_object_get_ex(gt_root, "images", &gt_images) && json_object_get_type(gt_images) == json_type_array) {
        int num_imgs = json_object_array_length(gt_images);
        for (int i = 0; i < num_imgs; ++i) {
            json_object* img = json_object_array_get_idx(gt_images, i);
            json_object *j_id, *j_file;
            if (json_object_object_get_ex(img, "id", &j_id) && json_object_object_get_ex(img, "file_name", &j_file)) {
                gt_filename_to_id[json_object_get_string(j_file)] = json_object_get_int(j_id);
            }
        }
    }

    // --- 3. Parse GT Annotations ---
    std::vector<GTAnnotation> gts;
    json_object* gt_annotations = nullptr;
    if (json_object_object_get_ex(gt_root, "annotations", &gt_annotations) && json_object_get_type(gt_annotations) == json_type_array) {
        int num_gt = json_object_array_length(gt_annotations);
        for (int i = 0; i < num_gt; ++i) {
            json_object* ann = json_object_array_get_idx(gt_annotations, i);
            if (!ann || json_object_get_type(ann) != json_type_object) continue;

            GTAnnotation gt;
            json_object *j_id, *j_img_id, *j_cat_id, *j_bbox, *j_iscrowd;
            
            json_object_object_get_ex(ann, "id", &j_id);
            json_object_object_get_ex(ann, "image_id", &j_img_id);
            json_object_object_get_ex(ann, "category_id", &j_cat_id);
            json_object_object_get_ex(ann, "bbox", &j_bbox);
            
            if (json_object_object_get_ex(ann, "iscrowd", &j_iscrowd)) {
                if (json_object_get_int(j_iscrowd) == 1) continue; 
            }

            if (!j_bbox || json_object_get_type(j_bbox) != json_type_array || json_object_array_length(j_bbox) != 4) continue;

            gt.id = json_object_get_int(j_id);
            gt.image_id = json_object_get_int(j_img_id);
            gt.category_id = json_object_get_int(j_cat_id);

            gt.bbox.x = json_object_get_double(json_object_array_get_idx(j_bbox, 0));
            gt.bbox.y = json_object_get_double(json_object_array_get_idx(j_bbox, 1));
            gt.bbox.w = json_object_get_double(json_object_array_get_idx(j_bbox, 2));
            gt.bbox.h = json_object_get_double(json_object_array_get_idx(j_bbox, 3));

            gts.push_back(gt);
        }
    }

    // --- 4. Parse Detection JSON ---
    json_object* det_root = json_object_from_file(det_file);
    if (!det_root) {
        std::cerr << "Error: Failed to parse Detection file.\n";
        json_object_put(gt_root);
        return 1;
    }

    std::map<int, std::string> det_id_to_filename;
    json_object* det_images = nullptr;
    if (json_object_object_get_ex(det_root, "images", &det_images) && json_object_get_type(det_images) == json_type_array) {
        int num_imgs = json_object_array_length(det_images);
        for (int i = 0; i < num_imgs; ++i) {
            json_object* img = json_object_array_get_idx(det_images, i);
            json_object *j_id, *j_file;
            if (json_object_object_get_ex(img, "id", &j_id) && json_object_object_get_ex(img, "file_name", &j_file)) {
                det_id_to_filename[json_object_get_int(j_id)] = json_object_get_string(j_file);
            }
        }
    }

    std::vector<DetAnnotation> dets;
    json_object* det_array = nullptr;
    if (json_object_get_type(det_root) == json_type_array) {
        det_array = det_root;
    } else if (json_object_get_type(det_root) == json_type_object) {
        if (!json_object_object_get_ex(det_root, "annotations", &det_array)) {
            json_object_object_get_ex(det_root, "predictions", &det_array);
        }
    }

    if (det_array && json_object_get_type(det_array) == json_type_array) {
        int num_det = json_object_array_length(det_array);
        for (int i = 0; i < num_det; ++i) {
            json_object* ann = json_object_array_get_idx(det_array, i);
            if (!ann || json_object_get_type(ann) != json_type_object) continue;

            DetAnnotation det;
            json_object *j_img_id, *j_cat_id, *j_bbox, *j_score;
            
            json_object_object_get_ex(ann, "image_id", &j_img_id);
            json_object_object_get_ex(ann, "category_id", &j_cat_id);
            json_object_object_get_ex(ann, "bbox", &j_bbox);
            json_object_object_get_ex(ann, "score", &j_score);

            if (!j_bbox || json_object_get_type(j_bbox) != json_type_array || json_object_array_length(j_bbox) != 4) continue;

            int original_det_id = json_object_get_int(j_img_id);
            
            if (det_id_to_filename.count(original_det_id)) {
                std::string filename = det_id_to_filename[original_det_id];
                if (gt_filename_to_id.count(filename)) {
                    det.image_id = gt_filename_to_id[filename];
                } else {
                    continue; 
                }
            } else {
                det.image_id = original_det_id;
            }

            det.category_id = json_object_get_int(j_cat_id);
            det.score = j_score ? json_object_get_double(j_score) : 0.0;
            det.bbox.x = json_object_get_double(json_object_array_get_idx(j_bbox, 0));
            det.bbox.y = json_object_get_double(json_object_array_get_idx(j_bbox, 1));
            det.bbox.w = json_object_get_double(json_object_array_get_idx(j_bbox, 2));
            det.bbox.h = json_object_get_double(json_object_array_get_idx(j_bbox, 3));

            dets.push_back(det);
        }
    }

    // --- 5. Evaluate mAP based strictly on parsed Categories ---
    std::vector<double> iou_thresholds = {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95};
    double mean_ap_all_classes = 0.0;
    int valid_classes = 0;

    std::cout << "\n--- Per-Class Average Precision (AP @ 0.50:0.95) ---\n";

    // Iterate over the official categories dictionary parsed from the GT JSON
    for (const auto& cat_pair : categories) {
        int cat_id = cat_pair.first;
        std::string cat_name = cat_pair.second;

        std::vector<GTAnnotation> cat_gts;
        for (const auto& gt : gts) if (gt.category_id == cat_id) cat_gts.push_back(gt);

        std::vector<DetAnnotation> cat_dets;
        for (const auto& det : dets) if (det.category_id == cat_id) cat_dets.push_back(det);

        // If a class has no GT and no Detections, it is skipped in standard COCO eval
        if (cat_gts.empty() && cat_dets.empty()) continue;

        std::sort(cat_dets.begin(), cat_dets.end(), [](const DetAnnotation& a, const DetAnnotation& b) {
            return a.score > b.score;
        });

        double ap_sum_over_ious = 0.0;

        for (double iou_thresh : iou_thresholds) {
            std::vector<int> tp(cat_dets.size(), 0);
            std::vector<int> fp(cat_dets.size(), 0);
            std::map<int, bool> gt_matched;

            for (size_t d = 0; d < cat_dets.size(); ++d) {
                const auto& det = cat_dets[d];
                double max_iou = -1.0;
                int best_gt_id = -1;

                for (const auto& gt : cat_gts) {
                    if (gt.image_id == det.image_id) {
                        double iou = calculate_iou(det.bbox, gt.bbox);
                        if (iou > max_iou) {
                            max_iou = iou;
                            best_gt_id = gt.id;
                        }
                    }
                }

                if (max_iou >= iou_thresh) {
                    if (!gt_matched[best_gt_id]) {
                        tp[d] = 1;
                        gt_matched[best_gt_id] = true;
                    } else {
                        fp[d] = 1; 
                    }
                } else {
                    fp[d] = 1; 
                }
            }

            double ap = compute_ap(tp, fp, cat_gts.size());
            ap_sum_over_ious += ap;
        }

        double class_map = ap_sum_over_ious / iou_thresholds.size();
        mean_ap_all_classes += class_map;
        valid_classes++;

        // Print AP for this specific class using the parsed name
        std::cout << std::left << std::setw(20) << cat_name 
                  << " (ID: " << std::setw(3) << cat_id << ") = " 
                  << std::fixed << std::setprecision(4) << class_map << "\n";
    }

    double final_map = (valid_classes == 0) ? 0.0 : (mean_ap_all_classes / valid_classes);

    std::cout << "--------------------------------------------------\n";
    std::cout << "Overall COCO mAP @ [IoU=0.50:0.95] = " << std::fixed << std::setprecision(4) << final_map << "\n\n";

    json_object_put(gt_root);
    json_object_put(det_root);

    return 0;
}