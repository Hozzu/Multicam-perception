#ifndef _DETRESULT_
#define _DETRESULT_

typedef struct{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    float id;
} DetResult;

inline float calculate_iou(const DetResult& box1, const DetResult& box2) {
    float inter_xmin = std::max(box1.xmin, box2.xmin);
    float inter_ymin = std::max(box1.ymin, box2.ymin);
    float inter_xmax = std::min(box1.xmax, box2.xmax);
    float inter_ymax = std::min(box1.ymax, box2.ymax);

    float inter_width = std::max(0.0f, inter_xmax - inter_xmin);
    float inter_height = std::max(0.0f, inter_ymax - inter_ymin);
    float inter_area = inter_width * inter_height;

    if (inter_area == 0.0f) {
        return 0.0f;
    }

    float box1_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin);
    float box2_area = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin);

    float union_area = box1_area + box2_area - inter_area;

    return inter_area / union_area;
}

inline std::vector<DetResult> apply_nms(std::vector<DetResult>& results, float score_threshold, float iou_threshold) {
    std::vector<DetResult> filtered_results;
    for (const auto& res : results) {
        if (res.score >= score_threshold) {
            filtered_results.push_back(res);
        }
    }

    std::sort(filtered_results.begin(), filtered_results.end(), [](const DetResult& a, const DetResult& b) {
        return a.score > b.score;
    });

    std::vector<DetResult> final_results;
    while (!filtered_results.empty()) {
        DetResult best_box = filtered_results.front();
        final_results.push_back(best_box);

        std::vector<DetResult> remaining_boxes;
        for (size_t i = 1; i < filtered_results.size(); ++i) {
            if (calculate_iou(best_box, filtered_results[i]) < iou_threshold) {
                remaining_boxes.push_back(filtered_results[i]);
            }
        }
        
        filtered_results = remaining_boxes;
    }

    results = final_results;
    return final_results;
}
#endif //_DETRESULT_