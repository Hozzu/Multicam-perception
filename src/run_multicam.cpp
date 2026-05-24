#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>
#include <memory>
#include <iomanip>
#include <time.h>

#include <jpeglib.h>
#include <dirent.h>
#include <unistd.h>
#include <eigen3/Eigen/Dense>
#include <json-c/json.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <engine_interface.hpp>
#include "hailo_post.hpp"
#include "maccel_post.hpp"
#include "tflite_post.hpp"
#include "preprocess.hpp"

static bool g_verbose = true;

namespace Params {
    // 1. 2D Detection & NMS
    double score_threshold = 0.001;
    double iou_threshold = 0.8;
    
    // 2. Tracker Lifecycle
    double init_track_score = 0.7;
    int tentative_hit_thresh = 7;
    int del_tentative_thresh = 1;
    int del_confirmed_thresh = 30;
    double invisible_penalty = 0.95; 
    
    // 3. Kalman Filter: Initial Covariance Matrix
    double cov_p_init = 0.2;
    double cov_v_init = 15.0;
    double cov_p_delay = 5.0;
    double cov_v_delay = 10.0;
    
    // 4. Kalman Filter: Process Noise (Q)
    double noise_a_base = 8.0;
    double noise_a_speed = 0.001;
    double noise_a_ped = 60.0;
    double noise_a_cone = 0.01;
    double noise_a_parked = 0.01;
    
    // 5. Dynamics, Speed Limits & Damping
    double damp_moving = 0.85;
    double damp_parked = 0.5;
    double max_speed_default = 30.0;
    double max_speed_ped = 1.5;
    double max_speed_bike = 15.0;
    double max_speed_static = 0.5;
    double static_speed_thresh = 0.3;
    
    // 6. Yaw (Heading) Update Mechanism
    double yaw_speed_thresh = 0.5;
    double yaw_alpha = 0.2;
    
    // 7. Kalman Filter: Measurement Noise (R)
    double var_depth_base = 1.0;
    double var_depth_scale = 8.0;
    double var_bearing_base = 0.02;
    double var_bearing_scale = 50.0;
    double conf_scale = 5.0;
    double proj_scale = 10.0;
    double diag_load = 0.01; 
    double ema_alpha = 0.1;  
    
    // 8. Track-to-Measurement Matching
    double dyn_match_base = 4.0;
    double dyn_match_dist_scale = 0.20;
    double dyn_match_ped_mult = 1.5;
    double maha_thresh = 12.0;
    double eval_maha_cov_add = 1.0; 
    
    // 9. 3D Projection, Truncation & Edge Cases
    double min_focal_length = 1.0;
    double truncation_margin_px = 0.0;
    double min_box_dim_px = 5.0;
    double depth_ground_weight = 15.0;
    double depth_blend_ratio = 1.0;
    double fov_edge_margin = 0.02;
    double depth_max = 160.0;
    double depth_min = 0.1;
    double fov_edge_penalty = 0.6;
    double dist_penalty_div = 100.0;
    double depth_ratio_max = 1.3;
    double depth_ratio_min = 0.5;
    
    // 10. 3D Weighted Box Fusion (Spatial Fusion)
    double wbf_dyn_base = 1.5;
    double wbf_dyn_dist_scale = 0.08;
    double wbf_dyn_ped_mult = 0.4;

    // 11. Temporal & Stability Bounds
    double dt_delay_thresh = 0.5;      
    double default_dt = 0.083;
    double horizon_z_thresh = -0.05;   
    double min_dist_penalty = 0.01;      
    double min_proj_confidence = 0.1; 
}

// ============================================================================
//                          GLOBAL STATISTICS & THRESHOLDS
// ============================================================================

static unsigned int num_preproces = 0;
static double sum_preprocess_time = 0.0;
static double sum_postprocess_time = 0.0;
static double sum_infer_time = 0.0;
static double sum_fusion_time = 0.0;

static unsigned int num_turnaround = 0;
static double sum_turnaround = 0.0;
static double max_turnaround = 0.0;
static double theoretical_max_time = 0.0;

static std::mutex in_preprocess_mutex;
static std::mutex in_postprocess_mutex;
static std::mutex in_fusion_mutex;

// Utility function to get precise thread execution time safely
static double get_thread_time_ms() {
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) {
        return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
    }
    return 0.0; 
}

// Reset global statistics for each evaluation pipeline execution
static void reset_global_stats() {
    num_preproces = 0;
    sum_preprocess_time = 0.0;
    sum_postprocess_time = 0.0;
    sum_infer_time = 0.0;
    sum_fusion_time = 0.0;
    num_turnaround = 0;
    sum_turnaround = 0.0;
    max_turnaround = 0.0;
}

// ============================================================================
//                    INTERNAL DATA STRUCTURES & THREAD POOL
// ============================================================================
namespace {

class ThreadPool {
public:
    ThreadPool(size_t num_threads, const std::vector<int>& cpus) : stop(false), active_tasks(0) {
        for(size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this, cpus] {
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                for(int cpu : cpus) { CPU_SET(cpu, &cpuset); }
                pthread_t current_thread = pthread_self();
                if(pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
                    // Fail silently to avoid console spam
                }

                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    if (task) {
                        task();
                    }
                    
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
            std::unique_lock<std::mutex> wait_lock(wait_mutex);
            active_tasks++;
        }
        condition.notify_one();
    }

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
        for(std::thread &worker: workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex, wait_mutex;
    std::condition_variable condition, wait_condition;
    bool stop;
    int active_tasks;
};

struct Object3D {
    int track_id;       
    int class_id;
    double x, y, z;     
    double w, l, h;     
    double vx, vy;      
    double yaw;         
    double score;
    double proj_confidence; 
    std::string attribute_name;
    int visibility_level; 
    Eigen::Vector3d ego_translation;
};

struct Transform {
    Eigen::Vector3d translation;
    Eigen::Quaterniond rotation;
};

// Utility for precisely interpolating ego-poses across unsynced sweeps
static Transform interpolate_transform(const Transform& t1, uint64_t time1, const Transform& t2, uint64_t time2, uint64_t target_time) {
    if (target_time <= time1) return t1;
    if (target_time >= time2) return t2;
    if (time1 == time2) return t1; 

    double ratio = static_cast<double>(target_time - time1) / static_cast<double>(time2 - time1);
    ratio = std::max(0.0, std::min(1.0, ratio));
    
    Transform interpolated;
    interpolated.translation = t1.translation + ratio * (t2.translation - t1.translation);
    
    // Ensure shortest path interpolation to prevent Gimbal lock / 360 spin
    Eigen::Quaterniond q2 = t2.rotation;
    if (t1.rotation.dot(q2) < 0.0) {
        q2.coeffs() = -q2.coeffs();
    }
    interpolated.rotation = t1.rotation.slerp(ratio, q2).normalized();
    return interpolated;
}

struct CamInfo {
    std::string path;
    Eigen::Matrix3d intrinsic;
    std::vector<double> distortion; 
    Transform cam_to_ego;
    Transform ego_to_global;
    int img_width, img_height;
};

struct MultiCamFrame {
    std::string sample_token;
    std::string scene_token; 
    uint64_t timestamp;
    bool is_key_frame;      
    std::vector<CamInfo> cameras;
    std::vector<Object3D> ground_truths; 
    Eigen::Vector3d sample_ego_translation;
    Eigen::Quaterniond sample_ego_rotation;
};

struct FrameEvalData {
    std::vector<Object3D> predictions;
    std::vector<Object3D> ground_truths;
    bool is_key_frame;
    Eigen::Vector3d ego_translation; 
    Eigen::Quaterniond ego_rotation;
};

struct TempCamData {
    uint64_t timestamp;
    std::string filename;
    bool is_key_frame;
    std::string sample_token;
    CamInfo info;
    int cam_idx;
};

struct TempAnn {
    std::string token;
    std::string sample_token;
    std::string instance_token;
    std::string prev;
    std::string next;
    uint64_t timestamp;
    Eigen::Vector3d pos;
    Object3D gt;
    int num_pts; 
};

struct EgoPoseRecord {
    uint64_t timestamp;
    Transform pose;
};

} // End of anonymous namespace

// ============================================================================
//                                UTILITY FUNCTIONS
// ============================================================================

// Official nuScenes 10 Classes Mapping & Priors (w, l, h)
static std::map<int, std::vector<double>> class_priors = {
    {0, {1.96, 4.63, 1.74}},  // Car
    {1, {2.51, 10.5, 3.47}},  // Truck
    {2, {2.94, 11.2, 3.47}},  // Bus
    {3, {2.90, 12.2, 3.80}},  // Trailer
    {4, {2.80, 6.40, 3.20}},  // Construction vehicle
    {5, {0.67, 0.73, 1.77}},  // Pedestrian
    {6, {0.77, 2.11, 1.47}},  // Motorcycle
    {7, {0.60, 1.70, 1.28}},  // Bicycle
    {8, {0.41, 0.41, 0.86}},  // Traffic cone
    {9, {2.50, 0.50, 0.98}}   // Barrier
};

static std::vector<double> get_prior(int class_id) {
    if (class_priors.find(class_id) != class_priors.end()) return class_priors[class_id];
    return {1.96, 4.63, 1.74}; 
}

static int coco_to_nuscenes(int coco_id) {
    switch(coco_id) {
        case 0: return 5; 
        case 1: return 7;
        case 2: return 0; 
        case 3: return 6; 
        case 5: return 2; 
        case 7: return 1; 
        default: return -1; 
    }
}

static int map_nuscenes_category(const std::string& cat) {
    if (cat == "vehicle.car") return 0;
    if (cat == "vehicle.truck") return 1;
    if (cat == "vehicle.bus.bendy" || cat == "vehicle.bus.rigid") return 2;
    if (cat == "vehicle.trailer") return 3;
    if (cat == "vehicle.construction") return 4;
    if (cat == "human.pedestrian.adult" || cat == "human.pedestrian.child" || 
        cat == "human.pedestrian.construction_worker" || cat == "human.pedestrian.police_officer") return 5;
    if (cat == "vehicle.motorcycle") return 6;
    if (cat == "vehicle.bicycle") return 7;
    if (cat == "movable_object.trafficcone") return 8;
    if (cat == "movable_object.barrier") return 9;
    return -1; 
}

static std::string get_nuscenes_class_name(int class_id) {
    switch(class_id) {
        case 0: return "car";
        case 1: return "truck";
        case 2: return "bus";
        case 3: return "trailer";
        case 4: return "construction_vehicle";
        case 5: return "pedestrian";
        case 6: return "motorcycle";
        case 7: return "bicycle";
        case 8: return "traffic_cone";
        case 9: return "barrier";
        default: return "car"; 
    }
}

static double bev_distance(const Object3D& a, const Object3D& b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

static double trunc_val(double val) {
    if (!std::isfinite(val)) return val;
    return std::round(val * 10000.0) / 10000.0;
}

// ============================================================================
//                               MULTI-OBJECT TRACKER
// ============================================================================

namespace {

enum TrackState {
    TENTATIVE = 0,
    CONFIRMED = 1,
    DELETED = 2
};

class MultiObjectTracker {
    struct Track {
        int id;
        int class_id;
        int invisible_count;
        int last_update_step; 
        int hit_count;
        TrackState state;
        Object3D obj;

        double current_step_weight_sum;
        double current_step_sum_sin_yaw;
        double current_step_sum_cos_yaw;

        Eigen::Vector4d statePost;
        Eigen::Matrix4d errorCovPost;
        Eigen::Matrix2d measurementNoiseCov;
        Eigen::Matrix<double, 2, 4> measurementMatrix;

        Track(Object3D init_obj, int track_id, int current_step) {
            id = track_id; 
            class_id = init_obj.class_id;
            invisible_count = 0; 
            last_update_step = current_step; 
            hit_count = 1; 
            
            state = (init_obj.score > Params::init_track_score) ? CONFIRMED : TENTATIVE; 

            obj = init_obj;
            obj.track_id = track_id; 
            
            current_step_weight_sum = obj.score * obj.proj_confidence;
            current_step_sum_sin_yaw = std::sin(obj.yaw) * current_step_weight_sum;
            current_step_sum_cos_yaw = std::cos(obj.yaw) * current_step_weight_sum;
            
            statePost << init_obj.x, init_obj.y, 0.0, 0.0;
            measurementMatrix << 1.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0;

            errorCovPost.setZero();
            errorCovPost.diagonal() << Params::cov_p_init, Params::cov_p_init, Params::cov_v_init, Params::cov_v_init; 
        }

        void predict(double dt) {
            if (!std::isfinite(dt) || dt <= 0.0) dt = 0.001; 
            if (dt > 2.0) dt = 2.0; 

            if (dt > Params::dt_delay_thresh) {
                errorCovPost.diagonal() << Params::cov_p_delay, Params::cov_p_delay, Params::cov_v_delay, Params::cov_v_delay; 
            }

            Eigen::Matrix4d transitionMatrix;
            transitionMatrix << 1.0, 0.0,  dt, 0.0,  
                                0.0, 1.0, 0.0,  dt,  
                                0.0, 0.0, 1.0, 0.0,   
                                0.0, 0.0, 0.0, 1.0;

            double current_speed = std::hypot(statePost(2), statePost(3));
            double noise_a = Params::noise_a_base + (current_speed * Params::noise_a_speed); 
            
            if (class_id == 5) noise_a = Params::noise_a_ped; 
            else if (class_id == 8 || class_id == 9) noise_a = Params::noise_a_cone; 
            else if (obj.attribute_name == "vehicle.parked") noise_a = Params::noise_a_parked; 

            double dt2 = dt * dt;
            double dt3 = dt2 * dt / 2.0;
            double dt4 = dt2 * dt2 / 4.0;
            
            Eigen::Matrix4d Q;
            Q << dt4 * noise_a, 0.0, dt3 * noise_a, 0.0,
                 0.0, dt4 * noise_a, 0.0, dt3 * noise_a,
                 dt3 * noise_a, 0.0, dt2 * noise_a, 0.0,
                 0.0, dt3 * noise_a, 0.0, dt2 * noise_a;

            statePost = transitionMatrix * statePost;
            errorCovPost = transitionMatrix * errorCovPost * transitionMatrix.transpose() + Q;

            double damp_factor = (obj.attribute_name == "vehicle.parked") ? Params::damp_parked : Params::damp_moving;
            statePost(2) *= damp_factor; 
            statePost(3) *= damp_factor;

            double max_speed = Params::max_speed_default; 
            if (class_id == 5) max_speed = Params::max_speed_ped; 
            else if (class_id == 7) max_speed = Params::max_speed_bike; 
            else if (class_id == 8 || class_id == 9) max_speed = Params::max_speed_static; 
            
            double speed = std::hypot(statePost(2), statePost(3));
            if (speed > max_speed && speed > 0.0001) {
                statePost(2) = (statePost(2) / speed) * max_speed;
                statePost(3) = (statePost(3) / speed) * max_speed;
            }

            if (std::isfinite(statePost(0)) && std::isfinite(statePost(1))) {
                obj.x = statePost(0); 
                obj.y = statePost(1);
                obj.vx = statePost(2); 
                obj.vy = statePost(3);
            }
            
            speed = std::hypot(obj.vx, obj.vy);
            
            if (speed > Params::yaw_speed_thresh && class_id >= 0 && class_id <= 4) { 
                double measured_yaw = std::atan2(obj.vy, obj.vx);
                double diff = measured_yaw - obj.yaw;
                
                while (diff > M_PI) diff -= 2.0 * M_PI;
                while (diff < -M_PI) diff += 2.0 * M_PI;
                
                if (std::abs(diff) > M_PI / 2.0) {
                    diff += (diff > 0.0) ? -M_PI : M_PI;
                }

                double alpha = Params::yaw_alpha; 
                obj.yaw = obj.yaw + alpha * diff;
                
                while (obj.yaw > M_PI) obj.yaw -= 2.0 * M_PI;
                while (obj.yaw < -M_PI) obj.yaw += 2.0 * M_PI;
            }

            if (dt > 0.0) invisible_count++;
        }

        void update(const Object3D& meas, int global_step) {
            if (!std::isfinite(meas.x) || !std::isfinite(meas.y)) return;

            if (this->last_update_step == global_step) {
                double weight_j = meas.score * meas.proj_confidence;
                double total_weight = this->current_step_weight_sum + weight_j;

                if (total_weight > 1e-06) {
                    this->obj.x = (this->obj.x * this->current_step_weight_sum + meas.x * weight_j) / total_weight;
                    this->obj.y = (this->obj.y * this->current_step_weight_sum + meas.y * weight_j) / total_weight;
                    this->obj.z = (this->obj.z * this->current_step_weight_sum + meas.z * weight_j) / total_weight;
                    
                    this->obj.w = (this->obj.w * this->current_step_weight_sum + meas.w * weight_j) / total_weight;
                    this->obj.l = (this->obj.l * this->current_step_weight_sum + meas.l * weight_j) / total_weight;
                    this->obj.h = (this->obj.h * this->current_step_weight_sum + meas.h * weight_j) / total_weight;
                    
                    double target_yaw = meas.yaw;
                    if (this->class_id >= 0 && this->class_id <= 4) {
                        if (std::cos(target_yaw - this->obj.yaw) < 0.0) {
                            target_yaw += M_PI;
                        }
                    }
                    
                    this->current_step_sum_sin_yaw += std::sin(target_yaw) * weight_j;
                    this->current_step_sum_cos_yaw += std::cos(target_yaw) * weight_j;
                    this->obj.yaw = std::atan2(this->current_step_sum_sin_yaw, this->current_step_sum_cos_yaw);
                }

                if (meas.score > this->obj.score) {
                    this->obj.ego_translation = meas.ego_translation;
                }
                this->obj.score = std::max(this->obj.score, meas.score);
                this->obj.proj_confidence = std::max(this->obj.proj_confidence, meas.proj_confidence);
                
                this->current_step_weight_sum = total_weight; 
                
                this->statePost(0) = this->obj.x;
                this->statePost(1) = this->obj.y;
                return;
            }

            this->last_update_step = global_step;
            this->hit_count++; 
            
            if (this->state == TENTATIVE && this->hit_count >= Params::tentative_hit_thresh) {
                this->state = CONFIRMED;
            }

            double dist_to_ego = std::hypot(meas.x - meas.ego_translation.x(), meas.y - meas.ego_translation.y());
            double angle_to_ego = std::atan2(meas.y - meas.ego_translation.y(), meas.x - meas.ego_translation.x());
            
            double c = std::cos(angle_to_ego);
            double s = std::sin(angle_to_ego);
            Eigen::Matrix2d R_rot;
            R_rot << c, -s, 
                     s,  c;
            
            double var_depth = Params::var_depth_base + std::pow(dist_to_ego / std::max(0.1, Params::var_depth_scale), 2.0); 
            double var_bearing = Params::var_bearing_base + (dist_to_ego / std::max(0.1, Params::var_bearing_scale));
            
            double conf_scale = 1.0 + (1.0 - meas.score) * Params::conf_scale;
            double proj_scale = 1.0 + (1.0 - meas.proj_confidence) * Params::proj_scale; 

            Eigen::Matrix2d R_diag;
            R_diag << (var_depth * conf_scale * proj_scale), 0.0,
                      0.0, (var_bearing * conf_scale * proj_scale);

            measurementNoiseCov = R_rot * R_diag * R_rot.transpose();

            Eigen::Vector2d measurement(meas.x, meas.y);

            Eigen::Matrix<double, 4, 2> H_T = measurementMatrix.transpose();
            Eigen::Matrix2d S = measurementMatrix * errorCovPost * H_T + measurementNoiseCov;
            
            double det = S.determinant();
            if (std::abs(det) < 1e-06 || !std::isfinite(det)) {
                S(0, 0) += Params::diag_load; S(1, 1) += Params::diag_load; 
            }

            Eigen::Matrix<double, 4, 2> K = errorCovPost * H_T * S.inverse();

            statePost = statePost + K * (measurement - measurementMatrix * statePost);
            Eigen::Matrix4d I = Eigen::Matrix4d::Identity();
            errorCovPost = (I - K * measurementMatrix) * errorCovPost;
            
            for (int i = 0; i < 4; ++i) {
                if (errorCovPost(i, i) > 1000.0) errorCovPost(i, i) = 1000.0;
                if (errorCovPost(i, i) < 0.01) errorCovPost(i, i) = 0.01;
                if (!std::isfinite(errorCovPost(i, i))) errorCovPost(i, i) = 10.0;
            }

            if (std::isfinite(statePost(0)) && std::isfinite(statePost(1))) {
                obj.x = statePost(0); 
                obj.y = statePost(1);
                obj.vx = statePost(2); 
                obj.vy = statePost(3);
            }
            
            double ema_alpha = Params::ema_alpha; 
            obj.score = (obj.score * (1.0 - ema_alpha)) + (meas.score * ema_alpha);
            
            obj.z = obj.z * (1.0 - ema_alpha) + meas.z * ema_alpha;
            obj.w = obj.w * (1.0 - ema_alpha) + meas.w * ema_alpha;
            obj.l = obj.l * (1.0 - ema_alpha) + meas.l * ema_alpha;
            obj.h = obj.h * (1.0 - ema_alpha) + meas.h * ema_alpha;
            
            obj.proj_confidence = meas.proj_confidence; 
            obj.ego_translation = meas.ego_translation; 
            invisible_count = 0;

            this->current_step_weight_sum = this->obj.score * this->obj.proj_confidence;
            this->current_step_sum_sin_yaw = std::sin(this->obj.yaw) * this->current_step_weight_sum;
            this->current_step_sum_cos_yaw = std::cos(this->obj.yaw) * this->current_step_weight_sum;
        }
    };

    std::vector<Track> tracks;
    int next_id = 1;
    int current_step = 0;
    std::mutex tracker_mutex;

    struct MatchPair {
        int trk_idx;
        int det_idx;
        double distance;
    };

    double calculate_mahalanobis(const Track& trk, const Object3D& det) {
        Eigen::Vector2d diff(det.x - trk.obj.x, det.y - trk.obj.y);
        Eigen::Matrix2d cov = trk.errorCovPost.block<2,2>(0,0);
        cov(0,0) += Params::eval_maha_cov_add; 
        cov(1,1) += Params::eval_maha_cov_add;
        
        double det_cov = cov.determinant();
        if (std::abs(det_cov) < 1e-06 || !std::isfinite(det_cov)) return 1e9; 
        
        double mahalanobis_sq = diff.transpose() * cov.inverse() * diff;
        return std::sqrt(std::max(0.0, mahalanobis_sq));
    }

public:
    void clear() {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        tracks.clear();
        next_id = 1;
        current_step = 0;
    }

    void predict_all(double dt) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        current_step++; 
        for (auto& trk : tracks) trk.predict(dt);
    }

    void update_measurements(const std::vector<Object3D>& detections) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        
        if (detections.empty()) return;
        if (tracks.empty()) {
            for (const auto& det : detections) {
                if(std::isfinite(det.x) && std::isfinite(det.y)) {
                    tracks.push_back(Track(det, next_id++, current_step));
                }
            }
            return;
        }

        std::vector<bool> det_matched(detections.size(), false);
        std::vector<bool> trk_matched(tracks.size(), false);
        std::vector<MatchPair> cost_matrix;

        for (size_t t = 0; t < tracks.size(); ++t) {
            double dist_to_ego = std::hypot(tracks[t].obj.x - tracks[t].obj.ego_translation.x(), 
                                            tracks[t].obj.y - tracks[t].obj.ego_translation.y());
            
            double dyn_match_dist = std::max(Params::dyn_match_base, dist_to_ego * Params::dyn_match_dist_scale); 
            if (tracks[t].class_id == 5) dyn_match_dist *= Params::dyn_match_ped_mult;

            for (size_t d = 0; d < detections.size(); ++d) {
                if (tracks[t].class_id != detections[d].class_id) continue;
                if (!std::isfinite(detections[d].x) || !std::isfinite(detections[d].y)) continue;
                
                double eucl_dist = bev_distance(tracks[t].obj, detections[d]);
                double maha_dist = calculate_mahalanobis(tracks[t], detections[d]);
                
                if (eucl_dist < dyn_match_dist && maha_dist < Params::maha_thresh) {
                    cost_matrix.push_back({static_cast<int>(t), static_cast<int>(d), maha_dist});
                }
            }
        }

        std::stable_sort(cost_matrix.begin(), cost_matrix.end(), [](const MatchPair& a, const MatchPair& b) {
            return a.distance < b.distance;
        });

        for (const auto& pair : cost_matrix) {
            if (!trk_matched[pair.trk_idx] && !det_matched[pair.det_idx]) {
                tracks[pair.trk_idx].update(detections[pair.det_idx], current_step);
                trk_matched[pair.trk_idx] = true;
                det_matched[pair.det_idx] = true;
            }
        }

        for (size_t d = 0; d < detections.size(); ++d) {
            if (!det_matched[d]) {
                if(std::isfinite(detections[d].x) && std::isfinite(detections[d].y)) {
                    tracks.push_back(Track(detections[d], next_id++, current_step));
                }
            }
        }
    }

    void cleanup_tracks() {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](Track& t) { 
            if (!std::isfinite(t.obj.x) || !std::isfinite(t.obj.y)) return true;
            
            if (t.state == TENTATIVE && t.invisible_count > Params::del_tentative_thresh) {
                t.state = DELETED;
            } else if (t.state == CONFIRMED && t.invisible_count > Params::del_confirmed_thresh) { 
                t.state = DELETED;
            }
            return t.state == DELETED;
        }), tracks.end());
    }

    std::vector<Object3D> get_active_tracks() {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        std::vector<Object3D> out;
        for (auto& trk : tracks) {
            if (trk.state == CONFIRMED) {
                Object3D obj = trk.obj;
                
                if (trk.invisible_count > 0) {
                    obj.score *= std::pow(Params::invisible_penalty, trk.invisible_count);
                }

                double speed = std::hypot(obj.vx, obj.vy);
                if (obj.class_id >= 0 && obj.class_id <= 4) { 
                    obj.attribute_name = (speed > Params::static_speed_thresh) ? "vehicle.moving" : "vehicle.parked";
                } else if (obj.class_id == 5) { 
                    obj.attribute_name = (speed > Params::static_speed_thresh) ? "pedestrian.moving" : "pedestrian.standing";
                } else if (obj.class_id == 6 || obj.class_id == 7) { 
                    obj.attribute_name = (speed > Params::static_speed_thresh) ? "cycle.with_rider" : "cycle.without_rider";
                } else {
                    obj.attribute_name = "";
                }
                
                trk.obj.attribute_name = obj.attribute_name; 
                out.push_back(obj);
            }
        }
        return out;
    }
};
} // End of anonymous namespace


// ============================================================================
//                               PIPELINE TASKS
// ============================================================================

static void preprocess_task(tflite::Interpreter * interpreter, int model_mode, std::string filename, int tensor_batch_idx, int cam_idx, std::vector<int> &img_heights, std::vector<int> &img_widths) {
    size_t last_slash_idx = filename.find_last_of("/\\");
    std::string base_filename = (last_slash_idx == std::string::npos) ? filename : filename.substr(last_slash_idx + 1);
    
    in_preprocess_mutex.lock();
    if (g_verbose) {
        std::cout << "Processing " << base_filename << "..\r";
        std::cout.flush();
    }
    in_preprocess_mutex.unlock();

    double preprocess_start = get_thread_time_ms();
    
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    FILE * fp = fopen(filename.c_str(), "rb");
    if(fp == NULL) {
        if (g_verbose) std::cerr << "WARNING: Cannot open the image: " << filename << ". Using default resolution fallback." << std::endl;
        jpeg_destroy_decompress(&cinfo);
        img_heights[cam_idx] = 900;
        img_widths[cam_idx] = 1600;
        return; 
    }
    jpeg_stdio_src(&cinfo, fp);
    
    if(jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        if (g_verbose) std::cerr << "WARNING: Invalid JPEG header for: " << filename << std::endl;
        fclose(fp);
        jpeg_destroy_decompress(&cinfo);
        img_heights[cam_idx] = 900;
        img_widths[cam_idx] = 1600;
        return;
    }

    cinfo.out_color_space = JCS_RGB;
    cinfo.output_components = 3;
    jpeg_start_decompress(&cinfo);

    int img_height = cinfo.output_height;
    int img_width = cinfo.output_width;
    int row_stride = cinfo.output_width * 3;

    if (img_height <= 0 || img_width <= 0) {
        if (g_verbose) std::cerr << "WARNING: JPEG dimensions invalid for: " << filename << std::endl;
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        img_heights[cam_idx] = 900;
        img_widths[cam_idx] = 1600;
        return;
    }

    img_heights[cam_idx] = img_height;
    img_widths[cam_idx] = img_width;

    uint8_t * rgb_buf_ptr = (uint8_t *)malloc(sizeof(uint8_t) * img_height * img_width * 3);
    if(rgb_buf_ptr == nullptr) {
        if (g_verbose) std::cerr << "ERROR: Memory allocation failed for image buffer.\n";
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        return;
    }

    while(cinfo.output_scanline < cinfo.output_height){
        uint8_t * rowptr = rgb_buf_ptr + row_stride * cinfo.output_scanline; 
        jpeg_read_scanlines(&cinfo, &rowptr, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    preprocess(interpreter, model_mode, rgb_buf_ptr, img_height, img_width, tensor_batch_idx);
    free(rgb_buf_ptr);

    double preprocess_elapsed = get_thread_time_ms() - preprocess_start;
    in_preprocess_mutex.lock();
    sum_preprocess_time += preprocess_elapsed;
    num_preproces++;
    in_preprocess_mutex.unlock();
}

static Object3D project_to_global_3d(const DetResult& box2d, const CamInfo& cam) {
    std::vector<double> prior = get_prior(box2d.id);
    double prior_w = prior[0], prior_l = prior[1], prior_h = prior[2];

    double safe_xmin = std::max(0.0, std::min(static_cast<double>(box2d.xmin), static_cast<double>(cam.img_width)));
    double safe_xmax = std::max(0.0, std::min(static_cast<double>(box2d.xmax), static_cast<double>(cam.img_width)));
    double safe_ymin = std::max(0.0, std::min(static_cast<double>(box2d.ymin), static_cast<double>(cam.img_height)));
    double safe_ymax = std::max(0.0, std::min(static_cast<double>(box2d.ymax), static_cast<double>(cam.img_height)));

    bool is_truncated_left = (safe_xmin <= Params::truncation_margin_px);
    bool is_truncated_right = (safe_xmax >= cam.img_width - Params::truncation_margin_px);
    bool is_truncated_top = (safe_ymin <= Params::truncation_margin_px);
    bool is_truncated_bottom = (safe_ymax >= cam.img_height - Params::truncation_margin_px);
    bool is_truncated_horizontal = is_truncated_left || is_truncated_right;

    double box_h_2d = std::max(Params::min_box_dim_px, safe_ymax - safe_ymin);
    double box_w_2d = std::max(Params::min_box_dim_px, safe_xmax - safe_xmin);

    double f_x = std::max(Params::min_focal_length, cam.intrinsic(0, 0));
    double f_y = std::max(Params::min_focal_length, cam.intrinsic(1, 1));

    double depth_h = (f_y * prior_h) / box_h_2d;
    double depth_w = (f_x * prior_w) / box_w_2d;

    double depth_prior = depth_h;

    if (is_truncated_top && !is_truncated_horizontal) {
        depth_prior = depth_w;
    } else if (is_truncated_horizontal && !is_truncated_top) {
        depth_prior = depth_h;
    } else if (box2d.id == 5 || box2d.id == 8 || box2d.id == 9) {
        double blend = Params::depth_blend_ratio;
        depth_prior = (depth_h * blend) + (depth_w * (1.0 - blend));
    }

    double u_center = (safe_xmin + safe_xmax) / 2.0;
    double v_center = (safe_ymin + safe_ymax) / 2.0;
    double u_bottom = u_center;
    double v_bottom = safe_ymax;

    if (!cam.distortion.empty() && cam.distortion.size() >= 4) {
        std::vector<cv::Point2f> src_pts = {cv::Point2f(u_bottom, v_bottom), cv::Point2f(u_center, v_center)};
        std::vector<cv::Point2f> dst_pts;
        
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
                                cam.intrinsic(0,0), cam.intrinsic(0,1), cam.intrinsic(0,2),
                                cam.intrinsic(1,0), cam.intrinsic(1,1), cam.intrinsic(1,2),
                                cam.intrinsic(2,0), cam.intrinsic(2,1), cam.intrinsic(2,2));
        
        cv::Mat distCoeffs = cv::Mat::zeros(1, cam.distortion.size(), CV_64F);
        for(size_t i = 0; i < cam.distortion.size(); ++i) distCoeffs.at<double>(0, i) = cam.distortion[i];

        cv::undistortPoints(src_pts, dst_pts, cameraMatrix, distCoeffs, cv::noArray(), cameraMatrix);
        
        u_bottom = dst_pts[0].x; v_bottom = dst_pts[0].y;
        u_center = dst_pts[1].x; v_center = dst_pts[1].y;
    }

    Eigen::Matrix3d safe_intrinsic = cam.intrinsic;
    if (std::abs(safe_intrinsic.determinant()) < 1e-06) {
        safe_intrinsic.setIdentity();
    }

    Eigen::Vector3d uv_center(u_center, v_center, 1.0);
    Eigen::Vector3d ray_cam_center = safe_intrinsic.inverse() * uv_center;
    ray_cam_center /= std::max(1e-06, ray_cam_center.z()); 
    Eigen::Vector3d P_cam_prior = ray_cam_center * depth_prior;
    Eigen::Vector3d P_ego_prior = cam.cam_to_ego.rotation * P_cam_prior + cam.cam_to_ego.translation;

    Eigen::Vector3d uv_bottom_vec(u_bottom, v_bottom, 1.0);
    Eigen::Vector3d ray_cam_bottom = safe_intrinsic.inverse() * uv_bottom_vec;
    ray_cam_bottom /= std::max(1e-06, ray_cam_bottom.z());
    Eigen::Vector3d ray_ego_bottom = cam.cam_to_ego.rotation * ray_cam_bottom;

    Eigen::Vector3d P_ego_ground_center = P_ego_prior;
    double depth_ground = -1.0;
    bool valid_ground = false;

    if (ray_ego_bottom.z() < Params::horizon_z_thresh) {
        double t_ground = -cam.cam_to_ego.translation.z() / ray_ego_bottom.z();
        if (t_ground > 0.0) {
            Eigen::Vector3d P_bottom_ego = cam.cam_to_ego.translation + t_ground * ray_ego_bottom;
            P_ego_ground_center = P_bottom_ego + Eigen::Vector3d(0.0, 0.0, prior_h / 2.0);
            depth_ground = t_ground * ray_cam_bottom.z();
            valid_ground = true;
        }
    }

    Eigen::Vector3d P_ego_final;
    double final_depth = depth_prior;

    if (!is_truncated_bottom && valid_ground && depth_ground > 0.0 && depth_ground < Params::dist_penalty_div) {
        double depth_ratio = depth_ground / std::max(0.1, depth_prior);

        if (depth_ratio > Params::depth_ratio_max || depth_ratio < Params::depth_ratio_min) {
            P_ego_final = P_ego_prior;
            final_depth = depth_prior;
        } else {
            double ground_weight = std::exp(-depth_ground / std::max(0.1, Params::depth_ground_weight));
            P_ego_final = (ground_weight * P_ego_ground_center) + ((1.0 - ground_weight) * P_ego_prior);
            final_depth = (ground_weight * depth_ground) + ((1.0 - ground_weight) * depth_prior);
        }
    } else {
        P_ego_final = P_ego_prior;
    }

    if (final_depth > Params::depth_max) final_depth = Params::depth_max;
    if (final_depth < Params::depth_min) final_depth = Params::depth_min;

    if (final_depth == Params::depth_max || final_depth == Params::depth_min) {
        P_cam_prior = ray_cam_center * final_depth;
        P_ego_final = cam.cam_to_ego.rotation * P_cam_prior + cam.cam_to_ego.translation;
    }

    Eigen::Vector3d pt_global = cam.ego_to_global.rotation * P_ego_final + cam.ego_to_global.translation;

    Object3D obj;
    obj.class_id = box2d.id;
    obj.x = pt_global.x();
    obj.y = pt_global.y();
    obj.z = pt_global.z();

    obj.w = prior_w; 
    obj.l = prior_l; 
    obj.h = prior_h;
    obj.vx = 0.0; 
    obj.vy = 0.0; 
    
    if (box2d.id >= 0 && box2d.id <= 4) {
        Eigen::Quaterniond q = cam.ego_to_global.rotation;
        double ego_yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()), 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
        obj.yaw = ego_yaw; 
    } else {
        obj.yaw = 0.0; 
    }
    
    double cx = cam.intrinsic(0, 2);
    double cy = cam.intrinsic(1, 2);
    double dist_from_center = std::hypot(u_center - cx, v_center - cy);
    double max_dist_from_center = std::max(1.0, std::hypot(cx, cy));
    
    double margin_ratio_x = std::min(safe_xmin, cam.img_width - safe_xmax) / cam.img_width;
    double edge_penalty = 1.0;
    if (margin_ratio_x < Params::fov_edge_margin || is_truncated_horizontal || is_truncated_bottom) {
        edge_penalty = Params::fov_edge_penalty; 
    }
    
    obj.proj_confidence = std::max(Params::min_proj_confidence, (1.0 - (dist_from_center / max_dist_from_center))) * edge_penalty;
    obj.attribute_name = ""; 

    double distance_penalty = std::max(Params::min_dist_penalty, 1.0 - (final_depth / std::max(0.1, Params::dist_penalty_div)));
    obj.score = box2d.score * distance_penalty;
    obj.visibility_level = 4;
    obj.ego_translation = cam.ego_to_global.translation;

    obj.x = trunc_val(obj.x);
    obj.y = trunc_val(obj.y);
    obj.z = trunc_val(obj.z);
    obj.w = trunc_val(obj.w);
    obj.l = trunc_val(obj.l);
    obj.h = trunc_val(obj.h);
    obj.yaw = trunc_val(obj.yaw);
    obj.score = trunc_val(obj.score);
    obj.proj_confidence = trunc_val(obj.proj_confidence);
    
    return obj;
}

static std::vector<Object3D> apply_3d_wbf(std::vector<Object3D>& all_3d) {
    if (all_3d.empty()) return {};

    std::stable_sort(all_3d.begin(), all_3d.end(), [](const Object3D& a, const Object3D& b) { 
        return (a.score * a.proj_confidence) > (b.score * b.proj_confidence); 
    });

    std::vector<Object3D> spatial_fused;
    std::vector<bool> merged(all_3d.size(), false);

    for (size_t i = 0; i < all_3d.size(); ++i) {
        if (merged[i]) continue;
        
        Object3D cluster = all_3d[i];
        double weight_i = cluster.score * cluster.proj_confidence;
        double total_score_weight = weight_i;
        
        double weighted_x = cluster.x * weight_i;
        double weighted_y = cluster.y * weight_i;
        double weighted_z = cluster.z * weight_i;
        
        double weighted_w = cluster.w * weight_i;
        double weighted_l = cluster.l * weight_i;
        double weighted_h = cluster.h * weight_i;
        
        double sum_sin_yaw = std::sin(cluster.yaw) * weight_i;
        double sum_cos_yaw = std::cos(cluster.yaw) * weight_i;
        double max_score = cluster.score;
        double max_proj_conf = cluster.proj_confidence;

        for (size_t j = i + 1; j < all_3d.size(); ++j) {
            if (merged[j]) continue;
            
            double dist_to_ego = std::hypot(all_3d[j].x - all_3d[j].ego_translation.x(), 
                                            all_3d[j].y - all_3d[j].ego_translation.y());
            
            double dyn_thresh = std::max(Params::wbf_dyn_base, dist_to_ego * Params::wbf_dyn_dist_scale);
            if (cluster.class_id == 5) dyn_thresh *= Params::wbf_dyn_ped_mult; 

            if (cluster.class_id == all_3d[j].class_id && bev_distance(cluster, all_3d[j]) < dyn_thresh) {
                double weight_j = all_3d[j].score * all_3d[j].proj_confidence;
                
                weighted_x += all_3d[j].x * weight_j;
                weighted_y += all_3d[j].y * weight_j;
                weighted_z += all_3d[j].z * weight_j;
                
                weighted_w += all_3d[j].w * weight_j;
                weighted_l += all_3d[j].l * weight_j;
                weighted_h += all_3d[j].h * weight_j;
                
                double target_yaw = all_3d[j].yaw;
                if (cluster.class_id >= 0 && cluster.class_id <= 4) {
                    if (std::cos(target_yaw - cluster.yaw) < 0.0) {
                        target_yaw += M_PI; 
                    }
                }
                
                sum_sin_yaw += std::sin(target_yaw) * weight_j;
                sum_cos_yaw += std::cos(target_yaw) * weight_j;

                total_score_weight += weight_j;

                if (all_3d[j].score > max_score) {
                    cluster.ego_translation = all_3d[j].ego_translation;
                }

                max_score = std::max(max_score, all_3d[j].score);
                max_proj_conf = std::max(max_proj_conf, all_3d[j].proj_confidence);
                
                merged[j] = true;
            }
        }
        
        if (total_score_weight > 1e-06) {
            cluster.x = weighted_x / total_score_weight;
            cluster.y = weighted_y / total_score_weight;
            cluster.z = weighted_z / total_score_weight;
            
            cluster.w = weighted_w / total_score_weight;
            cluster.l = weighted_l / total_score_weight;
            cluster.h = weighted_h / total_score_weight;
            
            cluster.yaw = std::atan2(sum_sin_yaw, sum_cos_yaw);
        }
        cluster.score = max_score; 
        cluster.proj_confidence = max_proj_conf;
        
        spatial_fused.push_back(cluster);
    }
    return spatial_fused;
}

static void simultaneous_batch_fusion_task(
    const std::vector<std::vector<DetResult>>& batch_boxes, 
    const std::vector<CamInfo>& batch_cams, 
    MultiObjectTracker& tracker) 
{
    double fusion_start = get_thread_time_ms();

    for (size_t c = 0; c < batch_boxes.size(); ++c) {
        std::vector<Object3D> local_3d;
        for (const auto& box : batch_boxes[c]) {
            if (box.score < Params::score_threshold) continue;
            if (box.id >= 0 && box.id <= 9) {
                local_3d.push_back(project_to_global_3d(box, batch_cams[c]));
            }
        }
        
        std::vector<Object3D> spatial_fused = apply_3d_wbf(local_3d);
        
        tracker.update_measurements(spatial_fused);
    }

    double fusion_elapsed = get_thread_time_ms() - fusion_start;
    in_fusion_mutex.lock();
    sum_fusion_time += fusion_elapsed;
    in_fusion_mutex.unlock();
}

static void sequential_fusion_task(std::vector<DetResult>& valid_results, const CamInfo& cam_info, MultiObjectTracker& tracker) {
    double fusion_start = get_thread_time_ms();
    
    std::vector<Object3D> local_3d;
    
    for (const auto& box : valid_results) {
        if (box.score < Params::score_threshold) continue;
        if (box.id >= 0 && box.id <= 9) {
            local_3d.push_back(project_to_global_3d(box, cam_info));
        }
    }
    
    std::vector<Object3D> spatial_fused = apply_3d_wbf(local_3d);
    tracker.update_measurements(spatial_fused);

    double fusion_elapsed = get_thread_time_ms() - fusion_start;
    in_fusion_mutex.lock();
    sum_fusion_time += fusion_elapsed;
    in_fusion_mutex.unlock();
}

static void postprocess_task_multicam(
    tflite::Interpreter * interpreter, int model_mode, 
    std::vector<int> &img_heights, std::vector<int> &img_widths, 
    int tensor_batch_idx, int cam_idx, 
    std::vector<DetResult> &out_results,
    std::string fusion_mode, const CamInfo& cam_info,
    MultiObjectTracker& tracker) 
{
    double postprocess_start = get_thread_time_ms();
    out_results.clear();

    float scale_height = 1.0f, scale_width = 1.0f;
    int input_height = 0, input_width = 0;
    std::vector<DetResult> results;

    int img_h_safe = std::max(img_heights[cam_idx], 1);
    int img_w_safe = std::max(img_widths[cam_idx], 1);

    if(interpreter->is_tflite_output(tensor_batch_idx)){
        TfLiteTensor* input_tensor_0 = interpreter->tflite_input_tensor(0);
        input_height = input_tensor_0->dims->data[1]; input_width = input_tensor_0->dims->data[2];
        scale_height = (float) input_height / img_h_safe;
        scale_width = (float) input_width / img_w_safe;
        if(model_mode != 1 && model_mode != 7) {
            float scale = scale_height < scale_width ? scale_height : scale_width;
            scale_height = scale; scale_width = scale;
        }
        if(model_mode != 6) tflite_post(interpreter, model_mode, results, true, tensor_batch_idx, Params::score_threshold, Params::iou_threshold);
    }
    else if(interpreter->is_hailo_output(tensor_batch_idx)){
        TfLiteTensor* input_tensor_0 = interpreter->hailo_input_tensor(0);
        input_height = input_tensor_0->dims->data[1]; input_width = input_tensor_0->dims->data[2];
        scale_height = (float) input_height / img_h_safe;
        scale_width = (float) input_width / img_w_safe;
        if(model_mode != 1 && model_mode != 7) {
            float scale = scale_height < scale_width ? scale_height : scale_width;
            scale_height = scale; scale_width = scale;
        }
        hailo_post(interpreter, model_mode, results, tensor_batch_idx, Params::score_threshold, Params::iou_threshold);
    }
    else if(interpreter->is_maccel_output(tensor_batch_idx)){
        TfLiteTensor* input_tensor_0 = interpreter->maccel_input_tensor(0);
        input_height = input_tensor_0->dims->data[1]; input_width = input_tensor_0->dims->data[2];
        scale_height = (float) input_height / img_h_safe;
        scale_width = (float) input_width / img_w_safe;
        if(model_mode != 1 && model_mode != 7) {
            float scale = scale_height < scale_width ? scale_height : scale_width;
            scale_height = scale; scale_width = scale;
        }
        maccel_post(interpreter, model_mode, results, true, tensor_batch_idx, Params::score_threshold, Params::iou_threshold);
    }

    std::vector<DetResult> valid_results;
    if (scale_width > 0.0f && scale_height > 0.0f) {
        for(int i = 0; i < results.size(); i++){
            if(results[i].score < Params::score_threshold) continue;
            DetResult res = results[i];
            
            res.xmin = res.xmin * input_width / scale_width; res.ymin = res.ymin * input_height / scale_height;
            res.xmax = res.xmax * input_width / scale_width; res.ymax = res.ymax * input_height / scale_height;

            if(model_mode == 4 || model_mode == 5) {
                if(res.id < 11) res.id = res.id;
                else if(res.id < 24) res.id = res.id + 1;
                else if(res.id < 26) res.id = res.id + 2;
                else if(res.id < 40) res.id = res.id + 4;
                else if(res.id < 60) res.id = res.id + 5;
                else if(res.id < 61) res.id = res.id + 6;
                else if(res.id < 62) res.id = res.id + 8;
                else if(res.id < 73) res.id = res.id + 9;
                else res.id = res.id + 10;
            }

            int nuscenes_id = coco_to_nuscenes(res.id);
            if (nuscenes_id == -1) continue; 
            res.id = nuscenes_id;
            valid_results.push_back(res);
        }
    }

    out_results = valid_results;

    double postprocess_elapsed = get_thread_time_ms() - postprocess_start;
    in_postprocess_mutex.lock();
    sum_postprocess_time += postprocess_elapsed;
    in_postprocess_mutex.unlock();

    if (fusion_mode == "sequential") {
        sequential_fusion_task(valid_results, cam_info, tracker);
    }
}


// ============================================================================
//                               JSON DATASET PARSER
// ============================================================================

static Eigen::Vector3d parse_vec3(json_object* arr) {
    if(!arr || json_object_array_length(arr) < 3) return Eigen::Vector3d::Zero();
    return Eigen::Vector3d(json_object_get_double(json_object_array_get_idx(arr, 0)),
                           json_object_get_double(json_object_array_get_idx(arr, 1)),
                           json_object_get_double(json_object_array_get_idx(arr, 2)));
}

static Eigen::Quaterniond parse_quat(json_object* arr) {
    if(!arr || json_object_array_length(arr) < 4) return Eigen::Quaterniond::Identity();
    return Eigen::Quaterniond(json_object_get_double(json_object_array_get_idx(arr, 0)), 
                              json_object_get_double(json_object_array_get_idx(arr, 1)), 
                              json_object_get_double(json_object_array_get_idx(arr, 2)), 
                              json_object_get_double(json_object_array_get_idx(arr, 3))).normalized();
}

static int extract_cam_idx(const std::string& filename) {
    if (filename.find("CAM_FRONT_LEFT") != std::string::npos) return 2;
    if (filename.find("CAM_FRONT_RIGHT") != std::string::npos) return 1;
    if (filename.find("CAM_FRONT") != std::string::npos) return 0;
    if (filename.find("CAM_BACK_LEFT") != std::string::npos) return 4;
    if (filename.find("CAM_BACK_RIGHT") != std::string::npos) return 5;
    if (filename.find("CAM_BACK") != std::string::npos) return 3;
    return -1;
}

static std::vector<MultiCamFrame> parse_nuscenes_dataset(std::string dataset_path) {
    std::string v1_mini_dir = dataset_path + "/v1.0-mini/";
    std::vector<MultiCamFrame> dataset;
    if (g_verbose) std::cout << "INFO: Parsing nuScenes database...\n";

    std::unordered_map<std::string, std::string> sensor_modality;
    json_object* sensor_root = json_object_from_file((v1_mini_dir + "sensor.json").c_str());
    if (sensor_root) {
        for (int i = 0; i < json_object_array_length(sensor_root); ++i) {
            json_object* item = json_object_array_get_idx(sensor_root, i);
            if(!item) continue;
            json_object *tok_obj = nullptr, *modality_obj = nullptr;
            if (json_object_object_get_ex(item, "token", &tok_obj) && json_object_object_get_ex(item, "modality", &modality_obj)) {
                sensor_modality[json_object_get_string(tok_obj)] = json_object_get_string(modality_obj);
            }
        }
        json_object_put(sensor_root);
    }

    std::unordered_map<std::string, bool> val_scene_tokens;
    json_object* scene_root = json_object_from_file((v1_mini_dir + "scene.json").c_str());
    if (scene_root) {
        for (int i = 0; i < json_object_array_length(scene_root); ++i) {
            json_object* item = json_object_array_get_idx(scene_root, i);
            if(!item) continue;
            json_object *tok_obj = nullptr, *name_obj = nullptr;
            if (json_object_object_get_ex(item, "token", &tok_obj) && json_object_object_get_ex(item, "name", &name_obj)) {
                std::string name = json_object_get_string(name_obj);
                if (name == "scene-0103" || name == "scene-0916") {
                    val_scene_tokens[json_object_get_string(tok_obj)] = true;
                }
            }
        }
        json_object_put(scene_root);
    }

    std::unordered_map<std::string, uint64_t> sample_times;
    std::unordered_map<std::string, bool> val_sample_tokens;
    std::unordered_map<std::string, std::string> sample_to_scene;
    json_object* sample_root = json_object_from_file((v1_mini_dir + "sample.json").c_str());
    if (sample_root) {
        for (int i = 0; i < json_object_array_length(sample_root); ++i) {
            json_object* item = json_object_array_get_idx(sample_root, i);
            if(!item) continue;
            json_object *tok_obj = nullptr, *time_obj = nullptr, *scene_tok_obj = nullptr;
            if (json_object_object_get_ex(item, "token", &tok_obj) && 
                json_object_object_get_ex(item, "timestamp", &time_obj) &&
                json_object_object_get_ex(item, "scene_token", &scene_tok_obj)) {
                
                std::string s_tok = json_object_get_string(tok_obj);
                std::string sc_tok = json_object_get_string(scene_tok_obj);
                sample_times[s_tok] = json_object_get_int64(time_obj);
                sample_to_scene[s_tok] = sc_tok;
                
                if (val_scene_tokens.find(sc_tok) != val_scene_tokens.end()) {
                    val_sample_tokens[s_tok] = true;
                }
            }
        }
        json_object_put(sample_root);
    }

    std::unordered_map<std::string, int> categories_map;
    json_object* cat_root = json_object_from_file((v1_mini_dir + "category.json").c_str());
    if (cat_root) {
        for (int i = 0; i < json_object_array_length(cat_root); ++i) {
            json_object* item = json_object_array_get_idx(cat_root, i);
            if(!item) continue;
            json_object *token_obj = nullptr, *name_obj = nullptr;
            if (json_object_object_get_ex(item, "token", &token_obj) && json_object_object_get_ex(item, "name", &name_obj)) {
                const char* token_str = json_object_get_string(token_obj);
                const char* name_str = json_object_get_string(name_obj);
                if(token_str && name_str) categories_map[token_str] = map_nuscenes_category(name_str);
            }
        }
        json_object_put(cat_root);
    }

    std::unordered_map<std::string, int> visibility_map;
    json_object* vis_root = json_object_from_file((v1_mini_dir + "visibility.json").c_str());
    if (vis_root) {
        for (int i = 0; i < json_object_array_length(vis_root); ++i) {
            json_object* item = json_object_array_get_idx(vis_root, i);
            if(!item) continue;
            json_object *token_obj = nullptr, *level_obj = nullptr;
            if (json_object_object_get_ex(item, "token", &token_obj) && json_object_object_get_ex(item, "level", &level_obj)) {
                std::string level_str = json_object_get_string(level_obj);
                int level_val = 4; 
                if (level_str == "v1") level_val = 1;
                else if (level_str == "v2") level_val = 2;
                else if (level_str == "v3") level_val = 3;
                else if (level_str == "v4") level_val = 4;
                visibility_map[json_object_get_string(token_obj)] = level_val;
            }
        }
        json_object_put(vis_root);
    }

    std::unordered_map<std::string, std::string> attribute_map;
    json_object* attr_root = json_object_from_file((v1_mini_dir + "attribute.json").c_str());
    if (attr_root) {
        for (int i = 0; i < json_object_array_length(attr_root); ++i) {
            json_object* item = json_object_array_get_idx(attr_root, i);
            if(!item) continue;
            json_object *token_obj = nullptr, *name_obj = nullptr;
            if (json_object_object_get_ex(item, "token", &token_obj) && json_object_object_get_ex(item, "name", &name_obj)) {
                attribute_map[json_object_get_string(token_obj)] = json_object_get_string(name_obj);
            }
        }
        json_object_put(attr_root);
    }

    std::unordered_map<std::string, int> instance_map;
    json_object* inst_root = json_object_from_file((v1_mini_dir + "instance.json").c_str());
    if (inst_root) {
        for (int i = 0; i < json_object_array_length(inst_root); ++i) {
            json_object* item = json_object_array_get_idx(inst_root, i);
            if(!item) continue;
            json_object *token_obj = nullptr, *cat_tok_obj = nullptr;
            if (json_object_object_get_ex(item, "token", &token_obj) && json_object_object_get_ex(item, "category_token", &cat_tok_obj)) {
                const char* inst_token_str = json_object_get_string(token_obj);
                const char* cat_token_str = json_object_get_string(cat_tok_obj);
                if (inst_token_str && cat_token_str && categories_map.find(cat_token_str) != categories_map.end()) {
                    instance_map[inst_token_str] = categories_map[cat_token_str];
                }
            }
        }
        json_object_put(inst_root);
    }

    std::unordered_map<std::string, CamInfo> calib_map;
    std::unordered_map<std::string, std::string> calib_to_modality;
    json_object* calib_root = json_object_from_file((v1_mini_dir + "calibrated_sensor.json").c_str());
    if (calib_root) {
        for (int i = 0; i < json_object_array_length(calib_root); ++i) {
            json_object* item = json_object_array_get_idx(calib_root, i);
            if(!item) continue;
            json_object *token_obj = nullptr, *sensor_tok_obj = nullptr, *trans_obj = nullptr, *rot_obj = nullptr, *intrin_obj = nullptr, *dist_obj = nullptr;
            if (json_object_object_get_ex(item, "token", &token_obj)) {
                const char* token_str = json_object_get_string(token_obj);
                if(!token_str) continue;
                
                if (json_object_object_get_ex(item, "sensor_token", &sensor_tok_obj)) {
                    const char* sens_tok = json_object_get_string(sensor_tok_obj);
                    if (sens_tok && sensor_modality.find(sens_tok) != sensor_modality.end()) {
                        calib_to_modality[token_str] = sensor_modality[sens_tok];
                    }
                }
                
                CamInfo info;
                if (json_object_object_get_ex(item, "camera_intrinsic", &intrin_obj) && json_object_array_length(intrin_obj) > 0) {
                    for (int r = 0; r < 3; r++) {
                        json_object* row = json_object_array_get_idx(intrin_obj, r);
                        for (int c = 0; c < 3; c++) info.intrinsic(r, c) = json_object_get_double(json_object_array_get_idx(row, c));
                    }
                } else { info.intrinsic = Eigen::Matrix3d::Identity(); }
                
                if (json_object_object_get_ex(item, "camera_distortion", &dist_obj) && json_object_array_length(dist_obj) > 0) {
                    for (int d = 0; d < json_object_array_length(dist_obj); d++) {
                        info.distortion.push_back(json_object_get_double(json_object_array_get_idx(dist_obj, d)));
                    }
                }
                
                if(json_object_object_get_ex(item, "translation", &trans_obj) && json_object_object_get_ex(item, "rotation", &rot_obj)){
                    info.cam_to_ego.translation = parse_vec3(trans_obj);
                    info.cam_to_ego.rotation = parse_quat(rot_obj);
                    calib_map[token_str] = info;
                }
            }
        }
        json_object_put(calib_root);
    }

    std::unordered_map<std::string, Transform> pose_map;
    std::vector<EgoPoseRecord> ego_pose_timeline;
    json_object* pose_root = json_object_from_file((v1_mini_dir + "ego_pose.json").c_str());
    if (pose_root) {
        for (int i = 0; i < json_object_array_length(pose_root); ++i) {
            json_object* item = json_object_array_get_idx(pose_root, i);
            if(!item) continue;
            json_object *token_obj = nullptr, *trans_obj = nullptr, *rot_obj = nullptr, *time_obj = nullptr;
            if (json_object_object_get_ex(item, "token", &token_obj) && json_object_object_get_ex(item, "timestamp", &time_obj)) {
                const char* token_str = json_object_get_string(token_obj);
                if(!token_str) continue;
                if(json_object_object_get_ex(item, "translation", &trans_obj) && json_object_object_get_ex(item, "rotation", &rot_obj)){
                    Transform t = {parse_vec3(trans_obj), parse_quat(rot_obj)};
                    pose_map[token_str] = t;
                    ego_pose_timeline.push_back({static_cast<uint64_t>(json_object_get_int64(time_obj)), t});
                }
            }
        }
        json_object_put(pose_root);
    }

    std::stable_sort(ego_pose_timeline.begin(), ego_pose_timeline.end(), [](const EgoPoseRecord& a, const EgoPoseRecord& b) {
        return a.timestamp < b.timestamp;
    });

    std::unordered_map<std::string, TempAnn> all_anns;
    json_object* ann_root = json_object_from_file((v1_mini_dir + "sample_annotation.json").c_str());
    if (ann_root) {
        for (int i = 0; i < json_object_array_length(ann_root); ++i) {
            json_object* item = json_object_array_get_idx(ann_root, i);
            if(!item) continue;
            json_object *tok_obj = nullptr, *samp_tok_obj = nullptr, *inst_tok_obj = nullptr, *prev_obj = nullptr, *next_obj = nullptr, *trans_obj = nullptr, *size_obj = nullptr, *rot_obj = nullptr, *nl_obj = nullptr, *nr_obj = nullptr, *vis_tok_obj = nullptr;

            if (json_object_object_get_ex(item, "token", &tok_obj) &&
                json_object_object_get_ex(item, "sample_token", &samp_tok_obj) &&
                json_object_object_get_ex(item, "instance_token", &inst_tok_obj)) {
                
                TempAnn ta;
                ta.token = json_object_get_string(tok_obj);
                ta.sample_token = json_object_get_string(samp_tok_obj);
                ta.instance_token = json_object_get_string(inst_tok_obj);
                
                if (json_object_object_get_ex(item, "prev", &prev_obj)) ta.prev = json_object_get_string(prev_obj);
                if (json_object_object_get_ex(item, "next", &next_obj)) ta.next = json_object_get_string(next_obj);
                
                ta.timestamp = sample_times[ta.sample_token]; 
                
                ta.num_pts = 0;
                if (json_object_object_get_ex(item, "num_lidar_pts", &nl_obj)) ta.num_pts += json_object_get_int(nl_obj);
                if (json_object_object_get_ex(item, "num_radar_pts", &nr_obj)) ta.num_pts += json_object_get_int(nr_obj);

                int class_id = -1;
                if (instance_map.find(ta.instance_token) != instance_map.end()) class_id = instance_map[ta.instance_token];
                
                if (class_id >= 0 && class_id <= 9 && json_object_object_get_ex(item, "translation", &trans_obj)) {
                    ta.gt.class_id = class_id;
                    std::hash<std::string> hasher;
                    ta.gt.track_id = hasher(ta.instance_token) % 10000;
                    ta.pos = parse_vec3(trans_obj);
                    ta.gt.x = ta.pos.x(); ta.gt.y = ta.pos.y(); ta.gt.z = ta.pos.z();

                    if (json_object_object_get_ex(item, "size", &size_obj)) {
                        ta.gt.w = json_object_get_double(json_object_array_get_idx(size_obj, 0));
                        ta.gt.l = json_object_get_double(json_object_array_get_idx(size_obj, 1));
                        ta.gt.h = json_object_get_double(json_object_array_get_idx(size_obj, 2));
                    }

                    if (json_object_object_get_ex(item, "rotation", &rot_obj)) {
                        Eigen::Quaterniond q = parse_quat(rot_obj);
                        ta.gt.yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()), 
                                               1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
                    }

                    json_object *attr_tok_obj = nullptr;
                    ta.gt.attribute_name = "";
                    if (json_object_object_get_ex(item, "attribute_tokens", &attr_tok_obj) && json_object_array_length(attr_tok_obj) > 0) {
                        std::string attr_tok = json_object_get_string(json_object_array_get_idx(attr_tok_obj, 0));
                        if (attribute_map.find(attr_tok) != attribute_map.end()) {
                            ta.gt.attribute_name = attribute_map[attr_tok];
                        }
                    }

                    ta.gt.visibility_level = 4;
                    if (json_object_object_get_ex(item, "visibility_token", &vis_tok_obj)) {
                        std::string v_tok = json_object_get_string(vis_tok_obj);
                        if (visibility_map.find(v_tok) != visibility_map.end()) {
                            ta.gt.visibility_level = visibility_map[v_tok];
                        }
                    }

                    all_anns[ta.token] = ta;
                }
            }
        }
        json_object_put(ann_root);
    }

    std::unordered_map<std::string, std::vector<Object3D>> gt_map;
    for (auto& kv : all_anns) {
        TempAnn& curr = kv.second;
        if (curr.num_pts == 0) continue; 
        if (val_sample_tokens.find(curr.sample_token) == val_sample_tokens.end()) continue;

        bool has_prev = !curr.prev.empty() && all_anns.find(curr.prev) != all_anns.end();
        bool has_next = !curr.next.empty() && all_anns.find(curr.next) != all_anns.end();

        if (!has_prev && !has_next) {
            curr.gt.vx = std::nan("");
            curr.gt.vy = std::nan("");
        } else {
            TempAnn* first = has_prev ? &all_anns[curr.prev] : &curr;
            TempAnn* last  = has_next ? &all_anns[curr.next] : &curr;

            double dt = (last->timestamp - first->timestamp) / 1e6;
            
            double max_time_diff = 1.5;
            if (has_prev && has_next) max_time_diff *= 2.0;

            if (dt > 0.0001 && dt <= max_time_diff) {
                curr.gt.vx = (last->pos.x() - first->pos.x()) / dt;
                curr.gt.vy = (last->pos.y() - first->pos.y()) / dt;
            } else {
                bool fallback_success = false;
                if (has_prev && has_next) {
                    double dt_fwd = (last->timestamp - curr.timestamp) / 1e6;
                    if (dt_fwd > 0.0001 && dt_fwd <= 1.5) {
                        curr.gt.vx = (last->pos.x() - curr.pos.x()) / dt_fwd;
                        curr.gt.vy = (last->pos.y() - curr.pos.y()) / dt_fwd;
                        fallback_success = true;
                    } else {
                        double dt_bwd = (curr.timestamp - first->timestamp) / 1e6;
                        if (dt_bwd > 0.0001 && dt_bwd <= 1.5) {
                            curr.gt.vx = (curr.pos.x() - first->pos.x()) / dt_bwd;
                            curr.gt.vy = (curr.pos.y() - first->pos.y()) / dt_bwd;
                            fallback_success = true;
                        }
                    }
                }
                if (!fallback_success) {
                    curr.gt.vx = std::nan("");
                    curr.gt.vy = std::nan("");
                }
            }
        }
        gt_map[curr.sample_token].push_back(curr.gt);
    }

    std::vector<TempCamData> cam_data_arrays[6]; 
    
    json_object* data_root = json_object_from_file((v1_mini_dir + "sample_data.json").c_str());
    if (data_root) {
        for (int i = 0; i < json_object_array_length(data_root); ++i) {
            json_object* item = json_object_array_get_idx(data_root, i);
            if(!item) continue;
            json_object *key_obj = nullptr, *samp_tok_obj = nullptr, *calib_tok_obj = nullptr, *file_obj = nullptr, *time_obj = nullptr;
            
            if (json_object_object_get_ex(item, "filename", &file_obj)) {
                const char* file_str = json_object_get_string(file_obj);
                if(!file_str) continue;
                std::string filename(file_str);

                if (json_object_object_get_ex(item, "calibrated_sensor_token", &calib_tok_obj)) {
                    const char* c_token = json_object_get_string(calib_tok_obj);
                    if (c_token && calib_to_modality.find(c_token) != calib_to_modality.end()) {
                        std::string modality = calib_to_modality[c_token];
                        if (modality != "camera") {
                            continue; 
                        }
                    } else continue; 
                } else continue;

                int cam_idx = extract_cam_idx(filename);
                if (cam_idx == -1) continue; 

                if (json_object_object_get_ex(item, "is_key_frame", &key_obj) &&
                    json_object_object_get_ex(item, "sample_token", &samp_tok_obj) &&
                    json_object_object_get_ex(item, "calibrated_sensor_token", &calib_tok_obj) &&
                    json_object_object_get_ex(item, "timestamp", &time_obj)) 
                {
                    const char* s_token = json_object_get_string(samp_tok_obj);
                    if (!s_token) continue;
                    
                    std::string scene_tok = sample_to_scene[s_token];
                    if (val_scene_tokens.find(scene_tok) == val_scene_tokens.end()) continue;

                    const char* c_token = json_object_get_string(calib_tok_obj);
                    if(!c_token) continue;

                    uint64_t frame_ts = json_object_get_int64(time_obj);

                    TempCamData td;
                    td.filename = filename;
                    td.cam_idx = cam_idx;
                    td.timestamp = frame_ts;
                    td.is_key_frame = json_object_get_boolean(key_obj); 
                    td.sample_token = s_token; 

                    CamInfo info = calib_map[c_token];
                    
                    Transform final_ego_pose;
                    if (ego_pose_timeline.empty()) {
                        final_ego_pose.translation = Eigen::Vector3d::Zero();
                        final_ego_pose.rotation = Eigen::Quaterniond::Identity();
                    } else {
                        auto it = std::lower_bound(ego_pose_timeline.begin(), ego_pose_timeline.end(), frame_ts,
                            [](const EgoPoseRecord& record, uint64_t ts) { return record.timestamp < ts; });
                        
                        if (it == ego_pose_timeline.end()) {
                            final_ego_pose = ego_pose_timeline.back().pose;
                        } else if (it == ego_pose_timeline.begin()) {
                            final_ego_pose = it->pose;
                        } else {
                            auto prev_it = it - 1;
                            final_ego_pose = interpolate_transform(prev_it->pose, prev_it->timestamp, it->pose, it->timestamp, frame_ts);
                        }
                    }
                    
                    info.ego_to_global = final_ego_pose;
                    info.path = dataset_path + std::string("/") + td.filename;
                    td.info = info;

                    cam_data_arrays[cam_idx].push_back(td);
                }
            }
        }
        json_object_put(data_root);
    }

    for (int i = 0; i < 6; ++i) {
        std::stable_sort(cam_data_arrays[i].begin(), cam_data_arrays[i].end(), [](const TempCamData& a, const TempCamData& b) {
            return a.timestamp < b.timestamp;
        });
    }

    for (const auto& front_td : cam_data_arrays[0]) {
        MultiCamFrame frame;
        frame.sample_token = front_td.sample_token;
        frame.scene_token = sample_to_scene[front_td.sample_token]; 
        frame.timestamp = front_td.timestamp;
        frame.is_key_frame = front_td.is_key_frame;
        frame.cameras.resize(6);
        frame.cameras[0] = front_td.info;
        
        frame.sample_ego_translation = front_td.info.ego_to_global.translation;
        frame.sample_ego_rotation = front_td.info.ego_to_global.rotation;
        
        if (frame.is_key_frame) {
            frame.ground_truths = gt_map[front_td.sample_token];
        }

        bool is_valid_sweep_cluster = true;
        for (int c = 1; c < 6; c++) {
            if (cam_data_arrays[c].empty()) {
                is_valid_sweep_cluster = false;
                break;
            }

            auto it = std::lower_bound(cam_data_arrays[c].begin(), cam_data_arrays[c].end(), front_td.timestamp, 
                [](const TempCamData& a, uint64_t ts) { return a.timestamp < ts; });

            TempCamData best_td;
            if (it == cam_data_arrays[c].end()) best_td = cam_data_arrays[c].back();
            else if (it == cam_data_arrays[c].begin()) best_td = *it;
            else {
                auto prev = it - 1;
                if ((it->timestamp - front_td.timestamp) < (front_td.timestamp - prev->timestamp)) best_td = *it;
                else best_td = *prev;
            }

            if (std::abs(static_cast<long long>(best_td.timestamp) - static_cast<long long>(front_td.timestamp)) > 250000) {
                is_valid_sweep_cluster = false;
            }
            frame.cameras[c] = best_td.info;
        }

        if (is_valid_sweep_cluster || frame.is_key_frame) {
            dataset.push_back(frame);
        }
    }

    std::stable_sort(dataset.begin(), dataset.end(), [](const MultiCamFrame& a, const MultiCamFrame& b) {
        return a.timestamp < b.timestamp;
    });

    return dataset;
}

static void build_json_annotations(const std::vector<Object3D>& tracked_3d, const MultiCamFrame& prev_frame, json_object* json_annotations) {
    if (prev_frame.is_key_frame) {
        // Create a local copy to sort and truncate without modifying the original tracking snapshot
        std::vector<Object3D> sorted_tracks = tracked_3d;
        
        // Sort tracks by score in descending order to preserve the most confident predictions
        std::stable_sort(sorted_tracks.begin(), sorted_tracks.end(), [](const Object3D& a, const Object3D& b) {
            return a.score > b.score;
        });
        
        // Enforce nuScenes devkit hard limit of maximum 500 boxes per sample
        size_t max_boxes = std::min(sorted_tracks.size(), static_cast<size_t>(500));
        
        for (size_t i = 0; i < max_boxes; ++i) {
            const auto& obj = sorted_tracks[i];
            json_object * ann = json_object_new_object();
            
            json_object_object_add(ann, "sample_token", json_object_new_string(prev_frame.sample_token.c_str()));
            
            json_object * trans = json_object_new_array();
            json_object_array_add(trans, json_object_new_double(std::isfinite(obj.x) ? obj.x : 0.0));
            json_object_array_add(trans, json_object_new_double(std::isfinite(obj.y) ? obj.y : 0.0));
            json_object_array_add(trans, json_object_new_double(std::isfinite(obj.z) ? obj.z : 0.0));
            json_object_object_add(ann, "translation", trans);
            
            json_object * size = json_object_new_array();
            json_object_array_add(size, json_object_new_double(std::isfinite(obj.w) ? std::max(0.01, obj.w) : 1.0));
            json_object_array_add(size, json_object_new_double(std::isfinite(obj.l) ? std::max(0.01, obj.l) : 1.0));
            json_object_array_add(size, json_object_new_double(std::isfinite(obj.h) ? std::max(0.01, obj.h) : 1.0));
            json_object_object_add(ann, "size", size);
            
            double safe_yaw = std::isfinite(obj.yaw) ? obj.yaw : 0.0;
            double half_yaw = safe_yaw * 0.5;
            json_object * rot = json_object_new_array();
            json_object_array_add(rot, json_object_new_double(std::cos(half_yaw))); 
            json_object_array_add(rot, json_object_new_double(0.0)); 
            json_object_array_add(rot, json_object_new_double(0.0)); 
            json_object_array_add(rot, json_object_new_double(std::sin(half_yaw))); 
            json_object_object_add(ann, "rotation", rot);

            json_object * vel = json_object_new_array();
            double safe_vx = std::isfinite(obj.vx) ? obj.vx : 0.0;
            double safe_vy = std::isfinite(obj.vy) ? obj.vy : 0.0;
            json_object_array_add(vel, json_object_new_double(safe_vx));
            json_object_array_add(vel, json_object_new_double(safe_vy));
            json_object_object_add(ann, "velocity", vel);
            
            json_object_object_add(ann, "detection_name", json_object_new_string(get_nuscenes_class_name(obj.class_id).c_str()));
            json_object_object_add(ann, "detection_score", json_object_new_double(obj.score));
            json_object_object_add(ann, "attribute_name", json_object_new_string(obj.attribute_name.c_str())); 
            
            in_fusion_mutex.lock();
            json_object_array_add(json_annotations, ann);
            in_fusion_mutex.unlock();
        }
    }
}

// ============================================================================
//                               NUSCENES EVALUATOR
// ============================================================================

class NuScenesEvaluator {
public:
    struct EvalMetrics {
        double mAP;
        double mATE;
        double mASE;
        double mAOE;
        double mAVE;
        double mAAE;
        double NDS;
    };

private:
    struct GlobalPred {
        int frame_idx;
        Object3D obj;
        int original_idx;
    };

    struct GlobalGT {
        Object3D obj;
        bool matched;
    };

    struct ClassMetrics {
        std::string name;
        double ap, ate, ase, aoe, ave, aae;
    };

    static double calculate_scale_error(const Object3D& a, const Object3D& b) {
        double intersection_vol = std::min(a.w, b.w) * std::min(a.l, b.l) * std::min(a.h, b.h);
        double union_vol = std::max(a.w, b.w) * std::max(a.l, b.l) * std::max(a.h, b.h);
        return 1.0 - (union_vol > 0.0 ? intersection_vol / union_vol : 0.0);
    }

    static double normalize_angle_diff(double p_yaw, double g_yaw, double period = 2.0 * M_PI) {
        double diff = p_yaw - g_yaw;
        diff = std::fmod(diff + period / 2.0, period);
        if (diff < 0.0) diff += period;
        diff -= period / 2.0;
        
        if (diff > M_PI) diff -= 2.0 * M_PI;
        else if (diff < -M_PI) diff += 2.0 * M_PI;
        
        return std::abs(diff);
    }

    static double np_interp(double x, const std::vector<double>& xp, const std::vector<double>& fp, bool use_right_val = false, double right_val = 0.0) {
        if (xp.empty()) return use_right_val ? right_val : 1.0; 
        if (x < xp.front()) return fp.front();
        if (x > xp.back()) return use_right_val ? right_val : fp.back();
        
        auto it = std::upper_bound(xp.begin(), xp.end(), x);
        if (it == xp.end()) return fp.back();
        
        size_t idx = std::distance(xp.begin(), it);
        double x0 = xp[idx - 1];
        double x1 = xp[idx];
        double y0 = fp[idx - 1];
        double y1 = fp[idx];
        
        if (x1 == x0) return y1; 
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
    }

    static std::vector<double> reverse_vec(const std::vector<double>& v) {
        std::vector<double> r = v;
        std::reverse(r.begin(), r.end());
        return r;
    }

public:
    static EvalMetrics evaluate_metrics(const std::vector<FrameEvalData>& evaluation_buffer, bool verbose = true) {
        if (verbose) {
            std::cout << "\n======================================================\n";
            std::cout << "  NuScenes Official Devkit Metrics Evaluation\n";
            std::cout << "======================================================\n";
        }

        std::vector<int> classes = {0, 1, 2, 5, 6, 7}; 
        std::vector<double> thresholds = {0.5, 1.0, 2.0, 4.0}; 

        double mean_ap = 0.0;
        double mean_ate = 0.0, mean_ase = 0.0, mean_aoe = 0.0, mean_ave = 0.0, mean_aae = 0.0;
        
        double count_aoe = 0.0, count_ave_aae = 0.0;
        std::vector<ClassMetrics> per_class_metrics;

        for (int cls : classes) {
            double class_ap_sum = 0.0;
            double c_ate = 1.0, c_ase = 1.0, c_aoe = 1.0, c_ave = 1.0, c_aae = 1.0; 

            double max_dist = 50.0; 
            if (cls == 5 || cls == 6 || cls == 7) max_dist = 40.0; 
            if (cls == 8 || cls == 9) max_dist = 30.0; 

            int total_gts = 0;
            std::vector<GlobalPred> all_preds;
            std::vector<std::vector<GlobalGT>> frame_gts(evaluation_buffer.size());

            int global_pred_counter = 0;

            for (size_t i = 0; i < evaluation_buffer.size(); ++i) {
                const auto& frame = evaluation_buffer[i];
                if (!frame.is_key_frame) continue;

                for (const auto& p : frame.predictions) {
                    if (p.class_id == cls) {
                        double dist_to_ego = std::hypot(p.x - frame.ego_translation.x(), p.y - frame.ego_translation.y());
                        if (dist_to_ego <= max_dist) {
                            GlobalPred gp; 
                            gp.frame_idx = i; 
                            gp.obj = p; 
                            gp.original_idx = global_pred_counter++;
                            all_preds.push_back(gp);
                        }
                    }
                }
                
                for (const auto& g : frame.ground_truths) {
                    if (g.class_id == cls) {
                        double dist_to_ego = std::hypot(g.x - frame.ego_translation.x(), g.y - frame.ego_translation.y());
                        if (dist_to_ego <= max_dist) {
                            GlobalGT gg; 
                            gg.obj = g; 
                            gg.matched = false;
                            frame_gts[i].push_back(gg);
                            total_gts++;
                        }
                    }
                }
            }

            std::stable_sort(all_preds.begin(), all_preds.end(), [](const GlobalPred& a, const GlobalPred& b) {
                if (a.obj.score != b.obj.score) return a.obj.score > b.obj.score;
                return a.original_idx > b.original_idx; 
            });

            for (double thresh : thresholds) {
                for (auto& gts : frame_gts) {
                    for (auto& gt : gts) gt.matched = false;
                }

                std::vector<double> tp(all_preds.size(), 0.0);
                std::vector<double> fp(all_preds.size(), 0.0);
                std::vector<double> conf(all_preds.size(), 0.0);
                
                std::vector<double> m_conf, m_ate, m_ase, m_aoe, m_ave, m_aae;

                for (size_t i = 0; i < all_preds.size(); ++i) {
                    int f_idx = all_preds[i].frame_idx;
                    double best_dist = 1e9;
                    int best_gt_idx = -1;

                    for (size_t j = 0; j < frame_gts[f_idx].size(); ++j) {
                        if (frame_gts[f_idx][j].matched) continue;
                        double dist = bev_distance(all_preds[i].obj, frame_gts[f_idx][j].obj);
                        if (dist < best_dist) { 
                            best_dist = dist; 
                            best_gt_idx = j; 
                        }
                    }

                    if (best_dist < thresh && best_gt_idx != -1) {
                        frame_gts[f_idx][best_gt_idx].matched = true;
                        tp[i] = 1.0;
                        const auto& gt_box = frame_gts[f_idx][best_gt_idx].obj;
                        
                        m_conf.push_back(all_preds[i].obj.score);
                        m_ate.push_back(best_dist);
                        m_ase.push_back(calculate_scale_error(all_preds[i].obj, gt_box));
                        
                        double eval_period = (cls == 9) ? M_PI : 2.0 * M_PI;
                        m_aoe.push_back(normalize_angle_diff(all_preds[i].obj.yaw, gt_box.yaw, eval_period));
                        
                        bool valid_vel = std::isfinite(all_preds[i].obj.vx) && std::isfinite(all_preds[i].obj.vy) &&
                                         std::isfinite(gt_box.vx) && std::isfinite(gt_box.vy);
                        if (valid_vel) {
                            m_ave.push_back(std::hypot(all_preds[i].obj.vx - gt_box.vx, all_preds[i].obj.vy - gt_box.vy));
                        } else {
                            m_ave.push_back(std::nan(""));
                        }

                        if (!all_preds[i].obj.attribute_name.empty() && 
                            all_preds[i].obj.attribute_name == gt_box.attribute_name) {
                            m_aae.push_back(0.0);
                        } else {
                            m_aae.push_back(1.0);
                        }
                    } else {
                        fp[i] = 1.0;
                    }
                    conf[i] = all_preds[i].obj.score;
                }

                double tp_sum = 0, fp_sum = 0;
                std::vector<double> prec(all_preds.size());
                std::vector<double> rec(all_preds.size());
                for (size_t i = 0; i < all_preds.size(); ++i) {
                    tp_sum += tp[i];
                    fp_sum += fp[i];
                    prec[i] = (tp_sum + fp_sum > 0) ? (tp_sum / (tp_sum + fp_sum)) : 0.0;
                    rec[i] = total_gts > 0 ? tp_sum / total_gts : 0.0;
                }

                std::vector<double> rec_interp(101);
                for (int i = 0; i < 101; ++i) rec_interp[i] = i / 100.0;
                
                std::vector<double> prec_interp(101, 0.0);
                std::vector<double> conf_interp(101, 0.0);
                
                for (int i = 0; i < 101; ++i) {
                    prec_interp[i] = np_interp(rec_interp[i], rec, prec, true, 0.0);
                    conf_interp[i] = np_interp(rec_interp[i], rec, conf, true, 0.0);
                }

                double min_recall = 0.1;
                double min_precision = 0.1;
                int first_ind = std::round(100.0 * min_recall) + 1;
                
                double ap = 0.0;
                for(int i = first_ind; i <= 100; ++i) {
                    double p = prec_interp[i] - min_precision;
                    if(p < 0.0) p = 0.0;
                    ap += p;
                }
                ap = (ap / 90.0) / (1.0 - min_precision);
                class_ap_sum += ap;

                if (thresh == 2.0) {
                    if (m_conf.empty()) {
                        c_ate = 1.0; c_ase = 1.0; c_aoe = 1.0; c_ave = 1.0; c_aae = 1.0;
                    } else {
                        auto apply_cummean = [](const std::vector<double>& vals) {
                            std::vector<double> res(vals.size(), 1.0);
                            bool all_nan = true;
                            for(double v : vals) if(std::isfinite(v)) { all_nan = false; break; }
                            if (all_nan) return res;

                            double sum = 0.0;
                            double valid_count = 0.0;
                            for (size_t i = 0; i < vals.size(); ++i) {
                                if (std::isfinite(vals[i])) {
                                    sum += vals[i];
                                    valid_count += 1.0;
                                }
                                res[i] = (valid_count > 0.0) ? (sum / valid_count) : std::nan("");
                            }
                            return res;
                        };

                        std::vector<double> ate_cum = apply_cummean(m_ate);
                        std::vector<double> ase_cum = apply_cummean(m_ase);
                        std::vector<double> aoe_cum = apply_cummean(m_aoe);
                        std::vector<double> ave_cum = apply_cummean(m_ave);
                        std::vector<double> aae_cum = apply_cummean(m_aae);

                        std::vector<double> rev_conf_interp = reverse_vec(conf_interp);
                        std::vector<double> rev_m_conf = reverse_vec(m_conf);
                        
                        auto interp_err = [&](const std::vector<double>& m_err) {
                            std::vector<double> rev_m_err = reverse_vec(m_err);
                            std::vector<double> res_rev(101);
                            for(int i = 0; i < 101; ++i) {
                                res_rev[i] = np_interp(rev_conf_interp[i], rev_m_conf, rev_m_err, false, 0.0);
                            }
                            return reverse_vec(res_rev);
                        };

                        std::vector<double> ate_interp = interp_err(ate_cum);
                        std::vector<double> ase_interp = interp_err(ase_cum);
                        std::vector<double> aoe_interp = interp_err(aoe_cum);
                        std::vector<double> ave_interp = interp_err(ave_cum);
                        std::vector<double> aae_interp = interp_err(aae_cum);

                        int max_recall_ind = 0;
                        for(int i = 0; i <= 100; ++i) {
                            if(conf_interp[i] > 0.0) max_recall_ind = i;
                        }

                        auto calc_tp_val = [&](const std::vector<double>& err_interp) {
                            if (max_recall_ind < first_ind) return 1.0;
                            double sum = 0.0;
                            for (int i = first_ind; i <= max_recall_ind; ++i) {
                                sum += err_interp[i];
                            }
                            return sum / (max_recall_ind - first_ind + 1.0);
                        };

                        c_ate = calc_tp_val(ate_interp);
                        c_ase = calc_tp_val(ase_interp);
                        c_aoe = calc_tp_val(aoe_interp);
                        c_ave = calc_tp_val(ave_interp);
                        c_aae = calc_tp_val(aae_interp);
                    }
                }
            }

            ClassMetrics cm;
            cm.name = get_nuscenes_class_name(cls);
            cm.ap = class_ap_sum / thresholds.size();
            cm.ate = c_ate;
            cm.ase = c_ase;
            cm.aoe = c_aoe;
            cm.ave = c_ave;
            cm.aae = c_aae;

            mean_ap += cm.ap;
            mean_ate += cm.ate; 
            mean_ase += cm.ase; 
            
            if (cls == 8) { 
                cm.aoe = std::nan(""); cm.ave = std::nan(""); cm.aae = std::nan("");
            } else if (cls == 9) { 
                cm.ave = std::nan(""); cm.aae = std::nan("");
                mean_aoe += cm.aoe; count_aoe += 1.0;
            } else {
                mean_aoe += cm.aoe; count_aoe += 1.0;
                mean_ave += cm.ave; mean_aae += cm.aae; count_ave_aae += 1.0;
            }
            per_class_metrics.push_back(cm);
        }

        double num_eval_classes = static_cast<double>(classes.size());
        
        mean_ap /= num_eval_classes; 
        mean_ate /= num_eval_classes; 
        mean_ase /= num_eval_classes;
        mean_aoe /= (count_aoe > 0.0 ? count_aoe : 1.0); 
        mean_ave /= (count_ave_aae > 0.0 ? count_ave_aae : 1.0); 
        mean_aae /= (count_ave_aae > 0.0 ? count_ave_aae : 1.0);

        // Official NuScenes NDS formula
        double tp_metrics_sum = std::max(1.0 - mean_ate, 0.0) + std::max(1.0 - mean_ase, 0.0) + 
                                std::max(1.0 - mean_aoe, 0.0) + std::max(1.0 - mean_ave, 0.0) + 
                                std::max(1.0 - mean_aae, 0.0);
                                
        double nds = (5.0 * mean_ap + tp_metrics_sum) / 10.0;

        if (verbose) {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "mAP:\t" << mean_ap << "\n";
            std::cout << "mATE:\t" << mean_ate << "\n";
            std::cout << "mASE:\t" << mean_ase << "\n";
            std::cout << "mAOE:\t" << mean_aoe << "\n";
            std::cout << "mAVE:\t" << mean_ave << "\n";
            std::cout << "mAAE:\t" << mean_aae << "\n";
            std::cout << "NDS:\t" << nds << "\n";
            std::cout << "------------------------------------------------------\n";
            
            std::cout << "Per-class results:\n";
            std::cout << std::left << std::setw(24) << "Object Class"
                      << std::setw(8) << "AP"
                      << std::setw(8) << "ATE"
                      << std::setw(8) << "ASE"
                      << std::setw(8) << "AOE"
                      << std::setw(8) << "AVE"
                      << std::setw(8) << "AAE" << "\n";
                      
            auto fmt_metric = [](double v) -> std::string {
                if (!std::isfinite(v)) return "nan";
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(3) << v;
                return oss.str();
            };

            for (const auto& cm : per_class_metrics) {
                std::cout << std::left << std::setw(24) << cm.name
                          << std::setw(8) << fmt_metric(cm.ap)
                          << std::setw(8) << fmt_metric(cm.ate)
                          << std::setw(8) << fmt_metric(cm.ase)
                          << std::setw(8) << fmt_metric(cm.aoe)
                          << std::setw(8) << fmt_metric(cm.ave)
                          << std::setw(8) << fmt_metric(cm.aae) << "\n";
            }
            std::cout << "\n";
        }
        
        EvalMetrics final_metrics;
        final_metrics.mAP = mean_ap;
        final_metrics.mATE = mean_ate;
        final_metrics.mASE = mean_ase;
        final_metrics.mAOE = mean_aoe;
        final_metrics.mAVE = mean_ave;
        final_metrics.mAAE = mean_aae;
        final_metrics.NDS = nds;
        
        return final_metrics;
    }
};

// ============================================================================
//                               MAIN EXECUTION
// ============================================================================

// Evaluates the full pipeline and returns the NDS evaluation.
double run_evaluation_pipeline(
    tflite::Interpreter* interpreter, 
    int model_mode, 
    std::vector<MultiCamFrame>& dataset, 
    const std::string& str_fusion_mode, 
    const std::vector<int>& exec_times, 
    int actual_batch_size, 
    bool save_results, 
    const char* result_path,
    bool verbose = true)
{
    g_verbose = verbose;
    reset_global_stats();
    
    MultiObjectTracker tracker; 
    ThreadPool pre_pool(6, {0, 1, 2, 3});
    ThreadPool post_pool(6, {0, 1, 2, 3});
    ThreadPool fusion_pool(1, {0, 1, 2, 3});

    std::vector<int> img_h[2], img_w[2];
    std::vector<std::vector<DetResult>> per_cam_2d_results[2];
    
    for(int k=0; k<2; k++) {
        img_h[k].resize(6); img_w[k].resize(6);
        per_cam_2d_results[k].resize(6);
    }

    int cur_buf = 0, prev_buf = 1;
    
    json_object * json_root = nullptr;
    json_object * json_results_dict = nullptr;
    
    if (save_results) {
        json_root = json_object_new_object();
        json_object * json_meta = json_object_new_object();
        json_object_object_add(json_meta, "use_camera", json_object_new_boolean(true));
        json_object_object_add(json_meta, "use_lidar", json_object_new_boolean(false));
        json_object_object_add(json_meta, "use_radar", json_object_new_boolean(false));
        json_object_object_add(json_meta, "use_map", json_object_new_boolean(false));
        json_object_object_add(json_meta, "use_external", json_object_new_boolean(false));
        json_object_object_add(json_root, "meta", json_meta);
        json_results_dict = json_object_new_object();
    }

    std::vector<FrameEvalData> evaluation_buffer;

    if (!dataset.empty()) {
        for (int c = 0; c < 6; c++) {
            pre_pool.enqueue([=, &img_h, &img_w, &dataset]() {
                preprocess_task(interpreter, model_mode, dataset[0].cameras[c].path, c, c, std::ref(img_h[cur_buf]), std::ref(img_w[cur_buf]));
            });
        }
        pre_pool.wait_all(); 
    }

    uint64_t prev_timestamp = 0;
    std::string prev_scene_token = "";

    auto app_start = std::chrono::steady_clock::now();

    for (size_t frame_idx = 0; frame_idx < dataset.size(); ++frame_idx) {
        MultiCamFrame* current = &dataset[frame_idx]; 
        MultiCamFrame* next = (frame_idx + 1 < dataset.size()) ? &dataset[frame_idx + 1] : nullptr; 

        fusion_pool.wait_all();

        if (current->scene_token != prev_scene_token) {
            tracker.clear();
            prev_scene_token = current->scene_token;
            prev_timestamp = current->timestamp;
        }

        double next_dt = Params::default_dt;
        if (frame_idx > 0) {
            next_dt = (current->timestamp - prev_timestamp) / 1e6;
            if (next_dt <= 0.0) next_dt = 0.001; 
            if (next_dt > 2.0) next_dt = 2.0; 
        }
        prev_timestamp = current->timestamp;

        double fusion_start = get_thread_time_ms();
        tracker.predict_all(next_dt);
        double fusion_elapsed = get_thread_time_ms() - fusion_start;
        
        in_fusion_mutex.lock();
        sum_fusion_time += fusion_elapsed;
        in_fusion_mutex.unlock();

        for(int c = 0; c < 6; c++) {
            current->cameras[c].img_width = img_w[cur_buf][c];
            current->cameras[c].img_height = img_h[cur_buf][c];
        }

        auto invoke_start = std::chrono::steady_clock::now();
        if(frame_idx == 0){
            if(interpreter->Invoke(true, exec_times) != kTfLiteOk) { break; }
        }
        else{
            if(interpreter->Invoke(true) != kTfLiteOk) { break; }
        }
        auto infer_elapsed = std::chrono::steady_clock::now() - invoke_start;
        sum_infer_time += std::chrono::duration_cast<std::chrono::microseconds>(infer_elapsed).count() / 1000.0;

        for(int c = 0; c < 6; c++) {
            post_pool.enqueue([=, &img_h, &img_w, &per_cam_2d_results, &tracker](){
                postprocess_task_multicam(
                    interpreter, model_mode, 
                    std::ref(img_h[cur_buf]), std::ref(img_w[cur_buf]), 
                    c, c, 
                    std::ref(per_cam_2d_results[cur_buf][c]),
                    str_fusion_mode, current->cameras[c],
                    tracker 
                );
            });
        }

        if (next) {
            for(int c = 0; c < 6; c++) {
                pre_pool.enqueue([=, &img_h, &img_w]() {
                    preprocess_task(interpreter, model_mode, next->cameras[c].path, c, c, std::ref(img_h[prev_buf]), std::ref(img_w[prev_buf]));
                });
            }
        }

        post_pool.wait_all();

        max_turnaround += interpreter->GetMaxTurnAroundTime();
        sum_turnaround += interpreter->GetSumTurnAroundTime();
        theoretical_max_time += interpreter->GetTheoreticalMaxTime();
        num_turnaround++;

        if (str_fusion_mode == "simultaneous") {
            std::vector<std::vector<DetResult>> batch_boxes;
            std::vector<CamInfo> batch_cams;
            for (int c = 0; c < 6; ++c) {
                batch_boxes.push_back(per_cam_2d_results[cur_buf][c]);
                batch_cams.push_back(current->cameras[c]);
            }
            fusion_pool.enqueue([batch_boxes, batch_cams, &tracker]() {
                simultaneous_batch_fusion_task(batch_boxes, batch_cams, tracker);
            });
        }

        if (next) pre_pool.wait_all();
        fusion_pool.wait_all();

        tracker.cleanup_tracks();
        std::vector<Object3D> tracks_snapshot_for_eval = tracker.get_active_tracks();
        
        FrameEvalData eval_data;
        eval_data.predictions = tracks_snapshot_for_eval;
        eval_data.ground_truths = current->ground_truths;
        eval_data.is_key_frame = current->is_key_frame;
        
        eval_data.ego_translation = current->sample_ego_translation;
        eval_data.ego_rotation = current->sample_ego_rotation; 
        
        if (current->is_key_frame) {
            evaluation_buffer.push_back(eval_data);
            
            if (save_results) {
                fusion_pool.enqueue([tracks_snapshot_for_eval, current_copy = *current, json_results_dict]() {
                    json_object* sample_array = json_object_new_array();
                    build_json_annotations(tracks_snapshot_for_eval, current_copy, sample_array);
                    
                    in_fusion_mutex.lock();
                    json_object* existing = nullptr;
                    if (!json_object_object_get_ex(json_results_dict, current_copy.sample_token.c_str(), &existing)) {
                        json_object_object_add(json_results_dict, current_copy.sample_token.c_str(), sample_array);
                    } else {
                        json_object_put(sample_array); 
                    }
                    in_fusion_mutex.unlock();
                });
            }
        }

        std::swap(cur_buf, prev_buf);
    }

    fusion_pool.wait_all();
    auto app_end_time = std::chrono::steady_clock::now();
    auto app_latency = std::chrono::duration_cast<std::chrono::milliseconds>(app_end_time - app_start).count();

    NuScenesEvaluator::EvalMetrics metrics = NuScenesEvaluator::evaluate_metrics(evaluation_buffer, verbose);

    if (save_results && verbose) {
        double avg_pre_per_data = (num_preproces > 0) ? (sum_preprocess_time / num_preproces) : 0.0;
        double avg_infer_per_data = (num_preproces > 0) ? (sum_infer_time / num_preproces) : 0.0;
        double avg_post_per_data = (num_preproces > 0) ? (sum_postprocess_time / num_preproces) : 0.0;
        double avg_fusion_per_data = (num_preproces > 0) ? (sum_fusion_time / num_preproces) : 0.0;
        double avg_turnaround_per_data = (num_turnaround > 0 && actual_batch_size > 0) ? ((sum_turnaround / num_turnaround) / actual_batch_size) : 0.0;
        double max_turnaround_per_batch = (num_turnaround > 0 ) ? (max_turnaround / num_turnaround) : 0.0;
        double max_turnaround_per_data = (num_turnaround > 0 ) ? (max_turnaround / num_turnaround / actual_batch_size) : 0.0;
        double theoretical_turnaround_per_batch = (num_turnaround > 0 ) ? (theoretical_max_time / num_turnaround) : 0.0;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Total items: " << num_preproces << "\n";
        std::cout << "Average preprocess time:\t" << avg_pre_per_data << " ms\n";
        std::cout << "Average infer time:\t\t" << avg_infer_per_data << " ms\n";
        std::cout << "Average postprocess time:\t" << avg_post_per_data << " ms\n";
        std::cout << "Average fusion time:\t\t" << avg_fusion_per_data << " ms\n";
        std::cout << "Average Turnaround time:\t" << avg_turnaround_per_data << " ms\n";
        std::cout << "Maximum Turnaround time:\t" << max_turnaround_per_batch << " ms\n";
        std::cout << "Theoretical Turnaround time:\t" << theoretical_turnaround_per_batch << " ms\n";
        std::cout << "Turnaround time for each data:\t" << max_turnaround_per_data << " ms\n"; 
        std::cout << "Application latency:\t\t" << app_latency / 1000.0 << " s\n";

        std::ofstream fout("result.txt", std::ios::app);
        fout << avg_turnaround_per_data << " " << max_turnaround_per_batch << " " << theoretical_turnaround_per_batch << " "
             << app_latency / 1000.0 << " "
             << metrics.mAP << " " << metrics.NDS << " "
             << metrics.mATE << " " << metrics.mASE << " " 
             << metrics.mAOE << " " << metrics.mAVE << " " << metrics.mAAE << std::endl;

        json_object_object_add(json_root, "results", json_results_dict);
        json_object_to_file(result_path, json_root);
        json_object_put(json_root);
    }
    
    return metrics.NDS;
}

bool run_multicam(tflite::Interpreter * interpreter, int model_mode, char * dataset_path, char * fusion_mode, char * result_path, std::vector<int> exec_times) {
    
    // Set CPU affinity to exclude cores 0-3
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
        return false;
    }
    
    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "ERROR: Failed to set thread affinity.\n";
        return false;
    }

    // Allocate tensors based on actual batch size requirement
    int actual_batch_size = 6;
    for(int i = 0; i < interpreter->inputs().size(); i++){
        TfLiteTensor* in_t = interpreter->input_tensor(i);
        if(interpreter->ResizeInputTensor(interpreter->inputs()[i], {actual_batch_size, in_t->dims->data[1], in_t->dims->data[2], in_t->dims->data[3]}) != kTfLiteOk){
            std::cerr << "ERROR: Hardware accelerator must support batch size 6 exactly.\n";
            return false;
        }
    }
    if(interpreter->AllocateTensors() != kTfLiteOk) return false;

    std::vector<MultiCamFrame> dataset = parse_nuscenes_dataset(std::string(dataset_path));
    if (dataset.empty()) {
        std::cerr << "ERROR: Dataset loading failed.\n"; return false;
    }

    std::string str_fusion_mode(fusion_mode);

    if(model_mode != 1){
        interpreter->SetScoreThreshold(static_cast<float>(Params::score_threshold));
        interpreter->SetIouThreshold(static_cast<float>(Params::iou_threshold));
    }

    std::cout << "[Run] Executing pipeline with optimal parameters & exporting results...\n";
    run_evaluation_pipeline(interpreter, model_mode, dataset, str_fusion_mode, exec_times, actual_batch_size, true, result_path, true);

    return true;
}