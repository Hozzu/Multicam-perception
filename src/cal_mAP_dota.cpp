#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <filesystem>
#include <json-c/json.h>

namespace fs = std::filesystem;

const bool DEBUG_MODE = false;

struct Point {
    double x, y;
};

struct GTAnnotation {
    int id;
    std::string base_name;
    int category_id;
    std::vector<Point> poly;
};

struct DetAnnotation {
    std::string base_name;
    int category_id;
    std::vector<Point> poly;
    double score;
};

// --- Geometry Helpers for OBB ---

double polygon_area(const std::vector<Point>& poly) {
    if (poly.size() < 3) return 0.0;
    double area = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        size_t j = (i + 1) % poly.size();
        area += (poly[i].x * poly[j].y - poly[j].x * poly[i].y);
    }
    return std::abs(area) / 2.0;
}

std::vector<Point> order_corners(std::vector<Point> pts) {
    if (pts.size() != 4) return pts;
    double cx = 0, cy = 0;
    for (const auto& p : pts) { cx += p.x; cy += p.y; }
    cx /= 4.0; cy /= 4.0;
    std::sort(pts.begin(), pts.end(), [cx, cy](const Point& a, const Point& b) {
        return std::atan2(a.y - cy, a.x - cx) < std::atan2(b.y - cy, b.x - cx);
    });
    return pts;
}

// [수정됨] Y-down 이미지 좌표계에 맞게 내부 판별 부호를 >= 0 으로 수정
bool is_inside(const Point& p, const Point& a, const Point& b) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) >= 0;
}

Point intersect(const Point& prev, const Point& curr, const Point& s, const Point& e) {
    double a1 = curr.y - prev.y, b1 = prev.x - curr.x, c1 = a1 * prev.x + b1 * prev.y;
    double a2 = e.y - s.y, b2 = s.x - e.x, c2 = a2 * s.x + b2 * s.y;
    double det = a1 * b2 - a2 * b1;
    if (std::abs(det) < 1e-9) return {0, 0};
    return {(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det};
}

std::vector<Point> clip_polygon(std::vector<Point> subject, const std::vector<Point>& clip) {
    for (size_t i = 0; i < clip.size(); ++i) {
        std::vector<Point> input = subject;
        subject.clear();
        if (input.empty()) break;
        Point s = clip[i], e = clip[(i + 1) % clip.size()];
        for (size_t j = 0; j < input.size(); ++j) {
            Point curr = input[j], prev = input[(j + input.size() - 1) % input.size()];
            if (is_inside(curr, s, e)) {
                if (!is_inside(prev, s, e)) subject.push_back(intersect(prev, curr, s, e));
                subject.push_back(curr);
            } else if (is_inside(prev, s, e)) {
                subject.push_back(intersect(prev, curr, s, e));
            }
        }
    }
    return subject;
}

double calculate_obb_iou(const std::vector<Point>& p1, const std::vector<Point>& p2) {
    double area1 = polygon_area(p1), area2 = polygon_area(p2);
    if (area1 <= 0 || area2 <= 0) return 0.0;
    double inter = polygon_area(clip_polygon(p1, p2));
    double union_area = area1 + area2 - inter;
    return (union_area > 0) ? (inter / union_area) : 0.0;
}

std::string get_base_name(const std::string& filename) {
    size_t lastdot = filename.find_last_of(".");
    return (lastdot == std::string::npos) ? filename : filename.substr(0, lastdot);
}

bool parse_poly(json_object* ann, const char* key, std::vector<Point>& out_poly) {
    json_object* j_arr = nullptr;
    if (!json_object_object_get_ex(ann, key, &j_arr)) return false;
    
    if (json_object_get_type(j_arr) == json_type_array && json_object_array_length(j_arr) == 8) {
        out_poly.clear();
        for (int i = 0; i < 4; ++i) {
            out_poly.push_back({json_object_get_double(json_object_array_get_idx(j_arr, i * 2)), 
                                json_object_get_double(json_object_array_get_idx(j_arr, i * 2 + 1))});
        }
        out_poly = order_corners(out_poly);
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    if (argc != 3) { 
        std::cerr << "Usage: " << argv[0] << " <gt_txt_dir> <det_json>\n"; 
        return 1; 
    }

    std::string gt_dir = argv[1];
    std::string det_file = argv[2];

    if (DEBUG_MODE) std::cout << "[DEBUG] Loading Detection JSON: " << det_file << "\n";

    json_object* det_root = json_object_from_file(det_file.c_str());
    if (!det_root) { std::cerr << "Failed to load Detection JSON file.\n"; return 1; }

    std::map<int, std::string> categories;
    json_object *j_cats, *j_imgs, *j_anns;
    
    if (json_object_object_get_ex(det_root, "categories", &j_cats)) {
        for (int i = 0; i < json_object_array_length(j_cats); ++i) {
            json_object* c = json_object_array_get_idx(j_cats, i);
            categories[json_object_get_int(json_object_object_get(c, "id"))] = json_object_get_string(json_object_object_get(c, "name"));
        }
    }

    std::map<std::string, std::pair<int, int>> image_dims;
    std::map<int, std::string> det_id_to_base_name;
    
    if (json_object_object_get_ex(det_root, "images", &j_imgs)) {
        for (int i = 0; i < json_object_array_length(j_imgs); ++i) {
            json_object* img = json_object_array_get_idx(j_imgs, i);
            int id = json_object_get_int(json_object_object_get(img, "id"));
            std::string fname = json_object_get_string(json_object_object_get(img, "file_name"));
            int w = json_object_get_int(json_object_object_get(img, "width"));
            int h = json_object_get_int(json_object_object_get(img, "height"));
            std::string base_name = get_base_name(fname);
            image_dims[base_name] = {w, h};
            det_id_to_base_name[id] = base_name;
        }
    }

    std::vector<DetAnnotation> dets;
    if (json_object_object_get_ex(det_root, "annotations", &j_anns)) {
        for (int i = 0; i < json_object_array_length(j_anns); ++i) {
            json_object* ann = json_object_array_get_idx(j_anns, i);
            DetAnnotation det;
            int d_img_id = json_object_get_int(json_object_object_get(ann, "image_id"));
            if (det_id_to_base_name.count(d_img_id)) {
                det.base_name = det_id_to_base_name[d_img_id];
            } else { continue; }
            det.category_id = json_object_get_int(json_object_object_get(ann, "category_id"));
            det.score = json_object_get_double(json_object_object_get(ann, "score"));
            
            if (parse_poly(ann, "poly", det.poly)) dets.push_back(det);
        }
    }

    if (DEBUG_MODE) std::cout << "[DEBUG] Parsed " << dets.size() << " detection boxes from JSON.\n";

    // Analyze GT classes to find offset
    int min_gt_class = 999999;
    int max_gt_class = -1;
    for (const auto& entry : fs::directory_iterator(gt_dir)) {
        if (entry.path().extension() == ".txt") {
            std::ifstream file(entry.path());
            int c_id; double dummy;
            while (file >> c_id >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy) {
                if (c_id < min_gt_class) min_gt_class = c_id;
                if (c_id > max_gt_class) max_gt_class = c_id;
            }
        }
    }
    
    // Auto-detect YOLO format (0-14) vs COCO (1-15)
    bool add_one_to_gt = false;
    if (min_gt_class == 0 || max_gt_class == 14) {
        add_one_to_gt = true;
    }

    if (DEBUG_MODE) {
        std::cout << "[DEBUG] GT class range found: " << min_gt_class << " to " << max_gt_class << "\n";
        std::cout << "[DEBUG] Applying +1 to GT classes? " << (add_one_to_gt ? "YES" : "NO") << "\n";
    }

    std::vector<GTAnnotation> gts;
    int gt_counter = 0;
    
    for (const auto& entry : fs::directory_iterator(gt_dir)) {
        if (entry.path().extension() == ".txt") {
            std::string base_name = entry.path().stem().string();
            if (image_dims.find(base_name) == image_dims.end()) continue;
            
            int img_w = image_dims[base_name].first;
            int img_h = image_dims[base_name].second;
            
            std::ifstream file(entry.path());
            int c_id; double x1, y1, x2, y2, x3, y3, x4, y4;
            
            while (file >> c_id >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4) {
                GTAnnotation gt;
                gt.id = ++gt_counter;
                gt.base_name = base_name;
                gt.category_id = add_one_to_gt ? (c_id + 1) : c_id;
                
                gt.poly.push_back({x1 * img_w, y1 * img_h});
                gt.poly.push_back({x2 * img_w, y2 * img_h});
                gt.poly.push_back({x3 * img_w, y3 * img_h});
                gt.poly.push_back({x4 * img_w, y4 * img_h});
                gt.poly = order_corners(gt.poly);
                gts.push_back(gt);
            }
        }
    }

    if (DEBUG_MODE) std::cout << "[DEBUG] Parsed " << gts.size() << " GT boxes from TXT files.\n";

    if (gts.empty()) {
        std::cerr << "Error: No valid GT boxes found.\n";
        return 1;
    }

    // Evaluate
    std::cout << "\n--- DOTA Evaluation Results (OBB) ---\n";
    double sum_ap_50 = 0, sum_map_coco = 0;
    int valid_cats = 0;
    int debug_iou_prints = 0;

    for (auto const& [cat_id, cat_name] : categories) {
        std::vector<GTAnnotation> c_gts;
        for (const auto& g : gts) if (g.category_id == cat_id) c_gts.push_back(g);
        
        std::vector<DetAnnotation> c_dets;
        for (const auto& d : dets) if (d.category_id == cat_id) c_dets.push_back(d);

        if (c_gts.empty() && c_dets.empty()) continue;
        
        std::sort(c_dets.begin(), c_dets.end(), [](const DetAnnotation& a, const DetAnnotation& b) { 
            return a.score > b.score; 
        });

        double ap_50 = 0, ap_coco_total = 0;
        
        for (int i = 0; i < 10; ++i) {
            double iou_thresh = 0.5 + i * 0.05;
            std::vector<int> tp(c_dets.size(), 0), fp(c_dets.size(), 0);
            std::map<int, bool> matched;

            for (size_t d = 0; d < c_dets.size(); ++d) {
                double max_iou = -1.0; int best_gt = -1;
                for (const auto& g : c_gts) {
                    if (g.base_name == c_dets[d].base_name) {
                        double iou = calculate_obb_iou(c_dets[d].poly, g.poly);
                        
                        // Debugging the first 5 potential matches
                        if (DEBUG_MODE && i == 0 && debug_iou_prints < 5) {
                            std::cout << "[DEBUG IoU] " << cat_name << " in " << g.base_name 
                                      << " | GT:(" << g.poly[0].x << "," << g.poly[0].y << ") Det:(" 
                                      << c_dets[d].poly[0].x << "," << c_dets[d].poly[0].y 
                                      << ") -> IoU: " << iou << "\n";
                            debug_iou_prints++;
                        }

                        if (iou > max_iou) { max_iou = iou; best_gt = g.id; }
                    }
                }
                if (max_iou >= iou_thresh && !matched[best_gt]) { tp[d] = 1; matched[best_gt] = true; }
                else fp[d] = 1;
            }

            double current_ap = 0;
            if (!c_gts.empty() && !tp.empty()) {
                std::vector<double> prec(tp.size()), rec(tp.size());
                int acc_tp = 0, acc_fp = 0;
                for (size_t k = 0; k < tp.size(); ++k) {
                    acc_tp += tp[k]; acc_fp += fp[k];
                    prec[k] = (double)acc_tp / (acc_tp + acc_fp);
                    rec[k] = (double)acc_tp / c_gts.size();
                }
                for (int k = 0; k <= 100; ++k) {
                    double t = k / 100.0, p_max = 0;
                    for (size_t j = 0; j < prec.size(); ++j) if (rec[j] >= t) p_max = std::max(p_max, prec[j]);
                    current_ap += p_max;
                }
                current_ap /= 101.0;
            }
            if (i == 0) ap_50 = current_ap;
            ap_coco_total += current_ap;
        }
        
        double ap_coco = ap_coco_total / 10.0;
        std::cout << std::left << std::setw(20) << cat_name 
                  << " | AP@50: " << std::fixed << std::setprecision(4) << ap_50 
                  << " | AP@50:95: " << ap_coco << "\n";
                  
        sum_ap_50 += ap_50; sum_map_coco += ap_coco; valid_cats++;
    }

    std::cout << "--------------------------------------------------\n";
    std::cout << "Overall DOTA mAP @ 0.50      : " << (valid_cats > 0 ? (sum_ap_50 / valid_cats) : 0.0) << "\n";
    std::cout << "Overall DOTA mAP @ 0.50:0.95 : " << (valid_cats > 0 ? (sum_map_coco / valid_cats) : 0.0) << "\n\n";

    json_object_put(det_root);
    return 0;
}