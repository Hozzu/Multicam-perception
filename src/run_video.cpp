// ==========================================
// File: run_video.cpp
// Description: Runs video inference and displays performance statistics.
// ==========================================

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <unistd.h>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <future>
#include <time.h>

#include <wayland-client.h>
#include <wayland-egl.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include <engine_interface.hpp>
#include "hailo_post.hpp"
#include "maccel_post.hpp"
#include "tflite_post.hpp"
#include "preprocess.hpp"

// --- Shared Global Variables ---
static float score_threshold = 0.4;

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

static int screen_width = 1920;
static int screen_height = 1080;

// --- Graphics and Display State Structure ---
struct DisplayState {
    wl_display *display = nullptr;
    wl_registry *registry = nullptr;
    wl_compositor *compositor = nullptr;
    wl_shell *shell = nullptr;
    wl_surface *surface = nullptr;
    wl_shell_surface *shell_surface = nullptr;
    wl_egl_window *egl_window = nullptr;

    EGLDisplay egl_display = EGL_NO_DISPLAY;
    EGLConfig egl_config = nullptr;
    EGLContext egl_context = EGL_NO_CONTEXT;
    EGLSurface egl_surface = EGL_NO_SURFACE;

    GLuint program = 0;
    GLuint texture_id = 0;
    GLint position_loc = -1;
    GLint texcoord_loc = -1;
    
    bool is_running = true;
};

// --- Wayland Callback Functions ---
static void shell_surface_ping(void *data, wl_shell_surface *shell_surface, uint32_t serial) {
    wl_shell_surface_pong(shell_surface, serial);
}
static void shell_surface_configure(void *data, wl_shell_surface *shell_surface, uint32_t edges, int32_t width, int32_t height) {}
static void shell_surface_popup_done(void *data, wl_shell_surface *shell_surface) {}

static const wl_shell_surface_listener shell_surface_listener = {
    .ping = shell_surface_ping,
    .configure = shell_surface_configure,
    .popup_done = shell_surface_popup_done
};

static void registry_global(void *data, wl_registry *registry, uint32_t name, const char *interface, uint32_t version) {
    DisplayState *d = static_cast<DisplayState*>(data);
    if (strcmp(interface, "wl_compositor") == 0) {
        d->compositor = (wl_compositor*)wl_registry_bind(registry, name, &wl_compositor_interface, 1);
    } else if (strcmp(interface, "wl_shell") == 0) {
        d->shell = (wl_shell*)wl_registry_bind(registry, name, &wl_shell_interface, 1);
    }
}
static void registry_global_remove(void *data, wl_registry *registry, uint32_t name) {}

static const wl_registry_listener registry_listener = {
    .global = registry_global,
    .global_remove = registry_global_remove
};

// --- OpenGL ES 2.0 Helper Functions ---
static const char *vertex_shader_source =
    "attribute vec4 a_position;   \n"
    "attribute vec2 a_texCoord;   \n"
    "varying vec2 v_texCoord;     \n"
    "void main()                  \n"
    "{                            \n"
    "   gl_Position = a_position; \n"
    "   v_texCoord = a_texCoord;  \n"
    "}                            \n";

static const char *fragment_shader_source =
    "precision mediump float;                            \n"
    "varying vec2 v_texCoord;                            \n"
    "uniform sampler2D s_texture;                        \n"
    "void main()                                         \n"
    "{                                                   \n"
    "  gl_FragColor = texture2D(s_texture, v_texCoord);  \n"
    "}                                                   \n";

static GLuint create_gles_program() {
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertex_shader_source, NULL);
    glCompileShader(vs);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragment_shader_source, NULL);
    glCompileShader(fs);

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

// --- Display System Initialization and Cleanup ---
static bool init_display_system(DisplayState &d, int width, int height) {
    d.display = wl_display_connect(NULL);
    if (!d.display) { std::cerr << "Failed to connect to Wayland display." << std::endl; return false; }

    d.registry = wl_display_get_registry(d.display);
    wl_registry_add_listener(d.registry, &registry_listener, &d);
    wl_display_dispatch(d.display);
    wl_display_roundtrip(d.display);
    if (!d.compositor || !d.shell) { std::cerr << "Failed to get Wayland compositor/shell." << std::endl; return false; }

    d.surface = wl_compositor_create_surface(d.compositor);
    d.shell_surface = wl_shell_get_shell_surface(d.shell, d.surface);
    wl_shell_surface_add_listener(d.shell_surface, &shell_surface_listener, &d);
    wl_shell_surface_set_toplevel(d.shell_surface);

    wl_surface_set_opaque_region(d.surface, NULL);
    wl_surface_set_input_region(d.surface, NULL);

    d.egl_window = wl_egl_window_create(d.surface, width, height);
    if (!d.egl_window) { std::cerr << "Failed to create EGL window." << std::endl; return false; }

    d.egl_display = eglGetDisplay((EGLNativeDisplayType)d.display);
    eglInitialize(d.egl_display, NULL, NULL);

    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };
    EGLint num_configs;
    eglChooseConfig(d.egl_display, config_attribs, &d.egl_config, 1, &num_configs);
    d.egl_surface = eglCreateWindowSurface(d.egl_display, d.egl_config, (EGLNativeWindowType)d.egl_window, NULL);
    
    EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    d.egl_context = eglCreateContext(d.egl_display, d.egl_config, EGL_NO_CONTEXT, context_attribs);
    eglMakeCurrent(d.egl_display, d.egl_surface, d.egl_surface, d.egl_context);

    d.program = create_gles_program();
    glUseProgram(d.program);
    d.position_loc = glGetAttribLocation(d.program, "a_position");
    d.texcoord_loc = glGetAttribLocation(d.program, "a_texCoord");

    glGenTextures(1, &d.texture_id);
    glBindTexture(GL_TEXTURE_2D, d.texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    std::vector<uint8_t> dummy_data(1 * 1 * 3, 0); 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, dummy_data.data());

    return true;
}

static void render_frame_on_screen(DisplayState &d, const cv::Mat &frame) {
    glBindTexture(GL_TEXTURE_2D, d.texture_id);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.data);

    int viewport_x, viewport_y, viewport_w, viewport_h;
    // Default to fullscreen width/height
    int target_w = screen_width;
    int target_h = screen_height;

    double video_aspect = (double)frame.cols / frame.rows;
    double target_aspect = (double)target_w / target_h;

    if (video_aspect > target_aspect) {
        viewport_w = target_w;
        viewport_h = static_cast<int>(target_w / video_aspect);
    } else {
        viewport_h = target_h;
        viewport_w = static_cast<int>(target_h * video_aspect);
    }

    // Centered offset is always 0 as fullscreen
    int target_x_offset = 0;
    viewport_x = target_x_offset + (target_w - viewport_w) / 2;
    viewport_y = (target_h - viewport_h) / 2;
    
    glViewport(0, 0, screen_width, screen_height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f); 
    glClear(GL_COLOR_BUFFER_BIT);

    glViewport(viewport_x, viewport_y, viewport_w, viewport_h);

    static const GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,  1.0f, 1.0f, 0.0f
    };
    
    glVertexAttribPointer(d.position_loc, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), vertices);
    glVertexAttribPointer(d.texcoord_loc, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), vertices + 2);
    glEnableVertexAttribArray(d.position_loc);
    glEnableVertexAttribArray(d.texcoord_loc);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    eglSwapBuffers(d.egl_display, d.egl_surface);
}

static void cleanup_display_system(DisplayState &d) {
    glDeleteProgram(d.program);
    glDeleteTextures(1, &d.texture_id);
    eglDestroySurface(d.egl_display, d.egl_surface);
    eglDestroyContext(d.egl_display, d.egl_context);
    eglTerminate(d.egl_display);
    wl_egl_window_destroy(d.egl_window);
    wl_shell_surface_destroy(d.shell_surface);
    wl_surface_destroy(d.surface);
    wl_compositor_destroy(d.compositor);
    wl_shell_destroy(d.shell);
    wl_registry_destroy(d.registry);
    wl_display_disconnect(d.display);
}


// --- ThreadPool Implementation ---
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
        for(std::thread &worker: workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    std::mutex wait_mutex;
    std::condition_variable wait_condition;
    int active_tasks;
};

// --- Preprocess and Postprocess Tasks ---
static void preprocess_task(tflite::Interpreter * interpreter, int model_mode, cv::Mat &frame, int cur_batch){
    double preprocess_start = get_thread_time_ms();

    uint8_t * rgb_buf_ptr = frame.data;
    int frame_height = frame.rows;
    int frame_width = frame.cols;

    preprocess(interpreter, model_mode, rgb_buf_ptr, frame_height, frame_width, cur_batch);

    double preprocess_elapsed = get_thread_time_ms() - preprocess_start;
    in_preprocess_mutex.lock();
    sum_preprocess_time += preprocess_elapsed;
    num_preproces++;
    in_preprocess_mutex.unlock();
}

static void postprocess_task(tflite::Interpreter * interpreter, int model_mode, std::vector<std::string> &labels, cv::Mat &cvimg, int cur_batch, std::shared_ptr<std::promise<void>> done_promise){
    double postprocess_start = get_thread_time_ms();

    int frame_height = cvimg.rows;
    int frame_width = cvimg.cols;

    // Postprocess
    if(interpreter->is_hailo_output(cur_batch)){
        // Get the input tensor size info
        TfLiteTensor* input_tensor_0 = interpreter->hailo_input_tensor(0);
        TfLiteIntArray* input_dims = input_tensor_0->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];

        // Calculate scale
        float scale_height = (float) input_height / frame_height;
        float scale_width = (float) input_width / frame_width;

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
                    if(score < score_threshold)
                        continue;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    int id = results[i].id;

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
                hailo_post(interpreter, model_mode, results, cur_batch);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    if(score < score_threshold)
                        continue;

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
    else if(interpreter->is_maccel_output(cur_batch)){
        // Get the input tensor size info
        TfLiteTensor* input_tensor_0 = interpreter->maccel_input_tensor(0);
        TfLiteIntArray* input_dims = input_tensor_0->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];

        // Calculate scale
        float scale_height = (float) input_height / frame_height;
        float scale_width = (float) input_width / frame_width;

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
            case 4: //yolo
            case 5: //yolov10
            {
                std::vector<DetResult> results;
                maccel_post(interpreter, model_mode, results, false, cur_batch);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    if(score < score_threshold)
                        continue;

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
                maccel_post(interpreter, model_mode, results, false, cur_batch);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    if(score < score_threshold)
                        continue;

                    int id = results[i].id;

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
            default:
            {
                std::cerr << "ERROR: Unsupported model mode in Maccel postprocess " << model_mode << "\n";
                return;
            }
        }
    }
    else if(interpreter->is_tflite_output(cur_batch)){
        // Get the input tensor size info
        TfLiteTensor* input_tensor_0 = interpreter->tflite_input_tensor(0);
        TfLiteIntArray* input_dims = input_tensor_0->dims;
        int input_height = input_dims->data[1];
        int input_width = input_dims->data[2];

        // Calculate scale
        float scale_height = (float) input_height / frame_height;
        float scale_width = (float) input_width / frame_width;

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
                tflite_post(interpreter, model_mode, results, false, cur_batch);

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
                tflite_post(interpreter, model_mode, results, false, cur_batch);

                for(int i = 0; i < results.size(); i++){
                    float score =  results[i].score;
                    int id = results[i].id;

                    if(score < score_threshold)
                        continue;

                    float xmin = results[i].xmin * input_width / scale_width;
                    float ymin = results[i].ymin * input_height / scale_height;
                    float xmax = results[i].xmax * input_width / scale_width;
                    float ymax = results[i].ymax * input_height / scale_height;

                    if(id < 11) id = id;
                    else if(id < 24) id = id + 1;
                    else if(id < 26) id = id + 2;
                    else if(id < 40) id = id + 4;
                    else if(id < 60) id = id + 5;
                    else if(id < 61) id = id + 6;
                    else if(id < 62) id = id + 8;
                    else if(id < 73) id = id + 9;
                    else id = id + 10;

                    char str[100];
                    sprintf(str, "class: %s, prob: %.1f", labels[id].c_str(), score);
                    cv::putText(cvimg, str, cv::Point(xmin, ymin), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0));
                    cv::rectangle(cvimg, cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax)), cv::Scalar(255, 0, 0), 2);
                }

                break;
            }
            case 6: // yolo obb
            {
                TfLiteTensor *output_tensor_0 = interpreter->tflite_output_tensor(0);
                TfLiteIntArray *output_dims = output_tensor_0->dims;
                int output_height = output_dims->data[1];
                int output_width = output_dims->data[2];

                float *output = interpreter->typed_tflite_output_tensor<float>(0, cur_batch);
                float(*output_arr)[output_width] = (float(*)[output_width])output;

                int max_detections = 300;
                float conf_threshold = score_threshold;
                float iou_threshold = 0.7;
                bool multi_label = false; 
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
                    if (max_score == 0 && cx == 0 && cy == 0 && width == 0 && height == 0 && angle == 0) break;
                    if (!multi_label && max_score > conf_threshold) {
                        scores.push_back(max_score); class_ids.push_back(max_class_id);
                        boxes.push_back(cv::RotatedRect(cv::Point2f(cx, cy), cv::Size2f(width, height), angle));
                    }
                }

                std::vector<int> nms_result;
                if (multi_label) {
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

    double postprocess_elapsed = get_thread_time_ms() - postprocess_start;
    in_postprocess_mutex.lock();
    sum_postprocess_time += postprocess_elapsed;
    num_postprocess++;
    in_postprocess_mutex.unlock();

    if(done_promise)
        done_promise->set_value();
}

// --- Inference Function ---
void infer(tflite::Interpreter * interpreter, int model_mode, std::vector<std::string> &labels, std::vector<cv::Mat> &preloaded_frames, int batch_size) {
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

    // Display Init
    DisplayState display_state;
    if (!init_display_system(display_state, screen_width, screen_height)) return;

    // --- PIPELINE SETUP ---
    ThreadPool pre_pool(4, {0, 1, 2, 3});
    ThreadPool post_pool(4, {0, 1, 2, 3});

    // Futures for tracking frame completion in the previous batch
    std::vector<std::future<void>> prev_post_futures;
    std::vector<int> prev_frame_indices;
    
    static int frame_counter = 0;
    static double fps = 0.0;
    static auto last_time = std::chrono::steady_clock::now();

    for (size_t batch_start_idx = 0; batch_start_idx < preloaded_frames.size() && display_state.is_running; batch_start_idx += batch_size) {
        int current_batch_sz = std::min((size_t)batch_size, preloaded_frames.size() - batch_start_idx);

        // 1. Start Preprocessing for Current Batch (N) - ASYNC
        for(int b = 0; b < current_batch_sz; b++) {
            size_t frame_idx = batch_start_idx + b;
            pre_pool.enqueue([=, &preloaded_frames]() {
                preprocess_task(interpreter, model_mode, std::ref(preloaded_frames[frame_idx]), b);
            });
        }

        // 2. Wait & Render Previous Batch (N-1) frame by frame
        //    This loop allows frames to be displayed as soon as they are ready,
        //    while maintaining the correct sequence.
        for(size_t i = 0; i < prev_post_futures.size(); ++i) {
            // Wait ONLY for the specific frame 'i' to be ready
            prev_post_futures[i].wait(); 

            // Render immediately
            int frame_idx = prev_frame_indices[i];
            cv::Mat &display_frame = preloaded_frames[frame_idx];

            frame_counter++;
            auto current_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = current_time - last_time;
            if(elapsed.count() > 1.0) {
                fps = frame_counter / elapsed.count();
                frame_counter = 0;
                last_time = current_time;
            }

            char fps_text[32];
            snprintf(fps_text, sizeof(fps_text), "FPS: %.2f", fps);

            cv::Point text_position;
            cv::Scalar text_color(0, 255, 0);
            int font_face = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 2.0;
            int thickness = 5;
            int margin = 50;

            // Default to left aligned text
            text_position = cv::Point(margin, 50);

            cv::putText(display_frame, fps_text, text_position, font_face, font_scale, text_color, thickness);

            render_frame_on_screen(display_state, display_frame);
            wl_display_dispatch_pending(display_state.display);
        }

        // Stats Update
        if(batch_start_idx != 0){
            max_turnaround += interpreter->GetMaxTurnAroundTime();
            sum_turnaround += interpreter->GetSumTurnAroundTime();
            num_turnaround++;
        }
        
        // 3. Wait for Preprocessing of Current Batch (N)
        pre_pool.wait_all();

        // 4. Invoke (N)
        auto infer_start = std::chrono::steady_clock::now();
        if(interpreter->Invoke(true, {194,1000000,53,48}) != kTfLiteOk){
            std::cerr << "ERROR: Model execute failed\n";
            break;
        }
        auto infer_elapsed = std::chrono::steady_clock::now() - infer_start;
        sum_infer_time += std::chrono::duration_cast<std::chrono::milliseconds>(infer_elapsed).count();
        num_infer++;

        // 5. Start Postprocessing for Current Batch (N)
        //    Prepare futures for the next iteration's render loop
        prev_post_futures.clear();
        prev_frame_indices.clear();
        
        for(int b = 0; b < current_batch_sz; b++) {
            size_t frame_idx = batch_start_idx + b;
            
            auto p_ptr = std::make_shared<std::promise<void>>();
            
            prev_post_futures.push_back(p_ptr->get_future());
            prev_frame_indices.push_back(frame_idx);

            post_pool.enqueue([=, &preloaded_frames, &labels]() {
                postprocess_task(interpreter, model_mode, labels, preloaded_frames[frame_idx], b, p_ptr);
            });
        }
    }

    // Flush final batch
    for(size_t i = 0; i < prev_post_futures.size(); ++i) {
        prev_post_futures[i].wait();
        render_frame_on_screen(display_state, preloaded_frames[prev_frame_indices[i]]);
        wl_display_dispatch_pending(display_state.display);
    }

    max_turnaround += interpreter->GetMaxTurnAroundTime();
    sum_turnaround += interpreter->GetSumTurnAroundTime();
    num_turnaround++;

    cleanup_display_system(display_state);
}

// --- Main Run Video ---
bool run_video(tflite::Interpreter * interpreter, int model_mode, std::vector<std::string> * labels_arg, char * video_path, int batch_size = 1) {
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

    // --- Video Decoding and Preloading ---
    std::vector<cv::Mat> preloaded_frames;
    int frame_width = 0, frame_height = 0;

    std::cout << "INFO: Starting video decoding (loading to memory)..." << std::endl;
    
    // --- Initialize GStreamer ---
    gst_init(nullptr, nullptr);
    GstElement *pipeline, *appsink;
    GError *error = nullptr;
    #if defined(__aarch64__)
        std::string pipeline_str = "filesrc location=" + std::string(video_path) + " ! qtdemux ! h264parse ! avdec_h264 ! queue ! videoconvert ! video/x-raw,format=RGB ! queue! appsink name=mysink sync=false drop=false";
    #elif defined(__x86_64__)
        std::string pipeline_str = "filesrc location=" + std::string(video_path) + " ! decodebin ! queue ! videoconvert ! video/x-raw,format=RGB ! queue ! appsink name=mysink sync=false drop=false";
    #endif
    pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline || error) {
        std::cerr << "GStreamer Error: " << (error ? error->message : "Failed to create pipeline") << std::endl;
        return false;
    }
    appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    g_object_set(G_OBJECT(appsink), "emit-signals", FALSE, "max-buffers", 1, "drop", TRUE, NULL);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    while (true) {
        GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
        if (!sample) {
            break; 
        }

        GstBuffer *buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;

        cv::Mat frame;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            GstCaps *caps = gst_sample_get_caps(sample);
            GstVideoInfo info;

            if (gst_video_info_from_caps(&info, caps)) {
                if (frame_width == 0 || frame_height == 0) {
                    frame_width = GST_VIDEO_INFO_WIDTH(&info);
                    frame_height = GST_VIDEO_INFO_HEIGHT(&info);
                }

                guint stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
                gsize offset = GST_VIDEO_INFO_PLANE_OFFSET(&info, 0);
                // Create mat header and CLONE data because GStreamer buffer will be unmapped
                cv::Mat header_mat(GST_VIDEO_INFO_HEIGHT(&info), GST_VIDEO_INFO_WIDTH(&info), CV_8UC3, map.data + offset, stride);
                frame = header_mat.clone();
            }
            gst_buffer_unmap(buffer, &map);
        }
        gst_sample_unref(sample);

        if (!frame.empty()) {
            preloaded_frames.push_back(frame);
        }
        
        // Optional: Limit frame count to prevent OOM
        if (preloaded_frames.size() >= 1500) break;
    }

    // Cleanup GStreamer immediately after loading
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsink);
    gst_object_unref(pipeline);
    std::cout << "INFO: Finished loading " << preloaded_frames.size() << " frames." << std::endl;

    if (preloaded_frames.empty()) {
        std::cerr << "ERROR: No frames decoded." << std::endl;
        return false;
    }

    // --- Inference & Display ---
    infer(interpreter, model_mode, std::ref(labels), std::ref(preloaded_frames), batch_size);

    auto application_elapsed = std::chrono::steady_clock::now() - application_start;
    auto application_latency = std::chrono::duration_cast<std::chrono::milliseconds>(application_elapsed).count();
    
    // Print final statistics
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

    return true;
}