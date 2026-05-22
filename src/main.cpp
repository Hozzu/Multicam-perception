#include <iostream>
#include <cstring>
#include <cstddef>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include <engine_interface.hpp>

// Existing function prototypes
bool run_qcarcam(tflite::Interpreter * interpreter, int model_mode, std::vector<std::string> * labels, char * display_path);
bool run_image(tflite::Interpreter * interpreter, int model_mode, std::vector<std::string> * labels, char * directory_path, char * result_path, int batch_size = 1);
bool run_video(tflite::Interpreter * interpreter, int model_mode, std::vector<std::string> * labels, char * video_path, int batch_size = 1);
bool run_demo(tflite::Interpreter * interpreter1, tflite::Interpreter * interpreter2, int model_mode, std::vector<std::string> * labels_arg, char * video_path, int batch_size = 1);
bool run_multicam(tflite::Interpreter * interpreter, int model_mode, char * dataset_path, char * fusion_mode, char * result_path, std::vector<int> exec_times);

int main(int argc, char * argv[]){
    int model_mode; // 1 for ssd_mobilenet
                    // 2 for efficientdet
                    // 3 for efficientdet-lite
                    // 4 for yolo
                    // 5 for yolov10
                    // 6 for yolo obb
                    // 7 for detr resnet
                    // 8 for rt-detr

    std::vector<std::string> labels;

    // Usage guide
    if(argc < 2 || strcmp(argv[1], "-help") == 0 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "--h") == 0){
        std::cout << "Usage: pkshin_detect camera [MODEL] [ACCELERATOR] [LABEL] [DISPLAY]\n";
        std::cout << "Camera mode runs the object detection using qcarcam API.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, DSP is supported.\n\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[DISPLAY] is path of the file defining the display setting.\n";
        
        std::cout << "Usage: pkshin_detect image [MODEL] [ACCELERATOR] [LABEL] [IMG_DIR] [RESULT]\n";
        std::cout << "Image mode runs the object detection with jpeg images.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, DSP is supported.\n\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[IMG_DIR] is path of the directory containing images.\n";
        std::cout << "[RESULT] is path of the result json file.\n";
        
        std::cout << "Usage: pkshin_detect video [MODEL] [ACCELERATOR] [LABEL] [VIDEO]\n";
        std::cout << "Video mode runs the object detection with a video file.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, DSP is supported.\n\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[VIDEO] is path of the video file.\n";

        std::cout << "Usage: pkshin_detect demo [MODEL] [ACCELERATOR] [LABEL] [VIDEO]\n";
        std::cout << "Demo mode runs the object detection with a video file.\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, DSP is supported.\n\n";
        std::cout << "[LABEL] is path of the label file.\n";
        std::cout << "[VIDEO] is path of the video file.\n";

        std::cout << "Usage: pkshin_detect multicam [MODEL] [ACCELERATOR] [DATASET_DIR] [FUSION_MODE] [RESULT]\n";
        std::cout << "Multicam mode runs 3D Object Detection using synchronized multi-camera dataset (e.g., nuScenes).\n";
        std::cout << "[MODEL] is path of the model file.\n";
        std::cout << "[ACCELERATOR] specifies the accelerator to run the inference. CPU, GPU, DSP is supported.\n";
        std::cout << "[DATASET_DIR] is the root directory of the nuScenes dataset (e.g., v1.0-mini).\n";
        std::cout << "[FUSION_MODE] is either 'sequential' or 'simultaneous' (Late Fusion strategy).\n";
        std::cout << "[RESULT] is path of the output 3D world representation json file.\n";
        return 0;
    }

    // Argument error checking
    if(strcmp(argv[1], "camera") != 0 && strcmp(argv[1], "image") != 0 && strcmp(argv[1], "video") != 0 && strcmp(argv[1], "demo") != 0 && strcmp(argv[1], "multicam") != 0){
        std::cerr << "ERROR: The fisrt argument must be either camera, image, video, demo, or multicam\n";
        std::cerr << "Type -help to see the guide\n";
        return 1;
    }

    if(strcmp(argv[1], "camera") == 0 && argc < 5){
        std::cerr << "ERROR: The camera mode requires at least 5 arguments\n";
        std::cerr << "Type -help to see the guide\n";
        return 1;
    }

    if(strcmp(argv[1], "image") == 0 && argc < 6){
        std::cerr << "ERROR: The image mode requires at least 6 arguments\n";
        std::cerr << "Type -help to see the guide\n";
        return 1;
    }

    if(strcmp(argv[1], "video") == 0 && argc < 5){
        std::cerr << "ERROR: The video mode requires at least 5 arguments\n";
        std::cerr << "Type -help to see the guide\n";
        return 1;
    }

    if(strcmp(argv[1], "demo") == 0 && argc < 5){
        std::cerr << "ERROR: The demo mode requires at least 5 arguments\n";
        std::cerr << "Type -help to see the guide\n";
        return 1;
    }

    if(strcmp(argv[1], "multicam") == 0 && argc < 7){
        std::cerr << "ERROR: The multicam mode requires at least 7 arguments\n";
        std::cerr << "Type -help to see the guide\n";
        return 1;
    }

    // Load the model and build the interpreter
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(argv[2]);
    if(model == NULL){
        std::cerr << "ERROR: Model load failed. Check the model name.\n";
        return 1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if(interpreter == NULL){
        std::cerr << "ERROR: Interpreter build failed.\n";
        return 1;
    }

    // Parse the model info
    std::string fileName = std::string(argv[2]);

    size_t lastSlashPos = fileName.find_last_of("/\\");
    if (lastSlashPos != std::string::npos) {
        fileName = fileName.substr(lastSlashPos + 1);
    }

    size_t lastDotPos = fileName.find_last_of('.');
    std::string fileNameWithoutExt = fileName;
    if (lastDotPos != std::string::npos) {
        fileNameWithoutExt = fileName.substr(0, lastDotPos);
    }

    if(fileNameWithoutExt.find("mobilenet") != std::string::npos && fileNameWithoutExt.find("ssd") != std::string::npos){
        std::cout << "INFO: Model file: ssd mobilenet\n";
        model_mode = 1;
    }
    else if(fileNameWithoutExt.find("efficientdet") != std::string::npos){
        if(fileNameWithoutExt.find("lite") != std::string::npos){
            std::cout << "INFO: Model file: efficientdet lite\n";
            model_mode = 3;
        }
        else{
            std::cout << "INFO: Model file: efficientdet\n";
            model_mode = 2;
        }
    }
    else if(fileNameWithoutExt.find("yolo") != std::string::npos){
        if(fileNameWithoutExt.find("obb") != std::string::npos){
            std::cout << "INFO: Model file: yolo obb\n";
            model_mode = 6;
        }
        else if(fileNameWithoutExt.find("v3") != std::string::npos){
            std::cout << "INFO: Model file: yolov3\n";
            model_mode = 4;
        }
        else if(fileNameWithoutExt.find("v5") != std::string::npos){
            std::cout << "INFO: Model file: yolov5\n";
            model_mode = 4;
        }
        else if(fileNameWithoutExt.find("v8") != std::string::npos){
                std::cout << "INFO: Model file: yolov8\n";
                model_mode = 4;
        }
        else if(fileNameWithoutExt.find("v9") != std::string::npos){
            std::cout << "INFO: Model file: yolov9\n";
            model_mode = 4;
        }
        else if(fileNameWithoutExt.find("v10") != std::string::npos){
            std::cout << "INFO: Model file: yolov10\n";
            model_mode = 5;
        }
        else if(fileNameWithoutExt.find("v11") != std::string::npos){
            std::cout << "INFO: Model file: yolov11\n";
            model_mode = 4;
        }
        else if(fileNameWithoutExt.find("v12") != std::string::npos){
            std::cout << "INFO: Model file: yolov12\n";
            model_mode = 4;
        }
        else{
            std::cerr << "ERROR: Check the model file name for yolo. Currently, yolov3, yolov5, yolov8, yolov8 obb, yolov9, yolov10, yolov11, yolov11 obb, yolov12 are only supported\n";
            return 1;
        }
    }
    else if(fileNameWithoutExt.find("detr") != std::string::npos){
        if(fileNameWithoutExt.find("resnet") != std::string::npos){
            std::cout << "INFO: Model file: detr resnet\n";
            model_mode = 7;
        }
        else if(fileNameWithoutExt.find("rt") != std::string::npos){
            std::cout << "INFO: Model file: rt detr\n";
            model_mode = 8;
        }
        else{

        }
    }
    else{
        std::cerr << "ERROR: Check the model file name. Currently, ssd mobilenet, efficientdet, efficientdet lite, yolov3, yolov5, yolov8, yolov8 obb, yolov9, yolov10, yolov11, yolov11 obb, yolov12, detr resnet, rt detr are only supported\n";
        return 1;
    }

// =====================================================================
//std::vector<int> dump_tensor_ids = {311, 344, 355, 358, 361, 364, 369, 421, 500, 503, 522, 523, 524, 525, 526, 527, 528, 531};
/*std::vector<int> dump_tensor_ids = {379, 381, 383, 454, 455, 472, 476, 477, 478, 479, 482, 483, 484, 488, 490, 702, 704, 744, 786, 790, 851, 867, 917, 983, 988, 999, 1097};
//std::vector<int> dump_tensor_ids = {471, 473, 475, 480, 493, 499, 558, 598, 622, 623, 624, 625, 627, 628, 887, 888, 889, 890, 891, 892, 893, 894, 895, 899, 901, 924, 929, 930, 945, 947, 985, 1054, 1062, 1063, 1064, 1072, 1074, 1075, 1078, 1081, 1082, 1085, 1086, 1087, 1088, 1089, 1090, 1102, 1100};
//std::vector<int> dump_tensor_ids = {1081, 1082, 1083, 1087, 1090, 1092, 1105, 1121, 1125, 1137, 1159, 1128, 1160, 1169, 1170, 1171, 1172, 1176, 1305, 1793, 2401, 3395, 3636, 3646, 3656, 3657, 3658, 3659, 3660, 3661, 3662};
//std::vector<int> dump_tensor_ids = {0, 159};
std::vector<int> current_outputs = interpreter->outputs();

for (int dump_id : dump_tensor_ids) {
    if (std::find(current_outputs.begin(), current_outputs.end(), dump_id) == current_outputs.end()) {
        current_outputs.push_back(dump_id);
    }
}

interpreter->SetOutputs(current_outputs);*/
// =====================================================================

    // Set the delegate
    std::string f_str(argv[2]);
    if (f_str.size() > 7 && f_str.substr(f_str.size() - 7) == ".tflite") {
        std::cout << "INFO: TFLite model detected.\n";
        if( strcmp(argv[3], "CPU") == 0 || strcmp(argv[3], "cpu") == 0){
            std::cout << "INFO: Run with CPU.\n";
        }
        else if( strcmp(argv[3], "GPU") == 0 || strcmp(argv[3], "gpu") == 0 ){
            TfLiteGpuDelegateOptionsV2 gpu_delegate_options = TfLiteGpuDelegateOptionsV2Default();
            gpu_delegate_options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
	        #if defined(__aarch64__)
                // SA8195P has a bug on OpenCL. Use OpenGL instead.
                gpu_delegate_options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
                std::cout << "INFO: ARM64 architecture detected. Enabling GL_ONLY flag for GPU delegate.\n";
            #endif

            auto * gpu_delegate_ptr = TfLiteGpuDelegateV2Create(&gpu_delegate_options);
            if(gpu_delegate_ptr == NULL){
                std::cerr << "ERROR: Cannot create gpu delegate.\n";
                return 1;
            }

            tflite::Interpreter::TfLiteDelegatePtr gpu_delegate(gpu_delegate_ptr, &TfLiteGpuDelegateV2Delete);

            if (interpreter->ModifyGraphWithDelegate(gpu_delegate.get()) != kTfLiteOk){
                std::cerr << "ERROR: Cannot convert model with gpu delegate\n";
                return 1;
            }

            std::cout << "INFO: Run with gpu delegate.\n";
        }
        else if( strcmp(argv[3], "dsp") == 0 || strcmp(argv[3], "DSP") == 0 ){
            TfLiteHexagonInitWithPath("/usr/lib");

            TfLiteHexagonDelegateOptions hexagon_options = TfLiteHexagonDelegateOptionsDefault();

            auto* npu_delegate_ptr = TfLiteHexagonDelegateCreate(&hexagon_options);
            if (npu_delegate_ptr == NULL) {
                std::cerr << "ERROR: Cannot create hexagon delegate. Check whether the hexagon library is in /usr/lib/.\n";
                return 1;
            }

            tflite::Interpreter::TfLiteDelegatePtr npu_delegate(npu_delegate_ptr, &TfLiteHexagonDelegateDelete);

            if(interpreter->ModifyGraphWithDelegate(npu_delegate.get()) != kTfLiteOk){
                std::cerr << "ERROR: Cannot convert model with hexagon delegate\n";
                return 1;
            }

            std::cout << "INFO: Run with hexagon delegate.\n";
        }
        else{
            std::cerr << "ERROR: the fourth argument must be either cpu, gpu, dsp.\n";
            return 1;
        }
    }

    if(interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "ERROR: Memory allocation for interpreter failed.\n";
        return 1;
    }

    // Set number of threads for each accelerator
    interpreter->SetNumThreads((sysconf(_SC_NPROCESSORS_ONLN) - 4) / 4);
    std::cout << "INFO: Set number of threads to " << (sysconf(_SC_NPROCESSORS_ONLN) - 4) / 4 << "\n";


    // Print input tensors
    for(int i = 0; i < interpreter->inputs().size(); i++){
        std::string input_dims_str("[");
        input_dims_str += std::to_string(interpreter->input_tensor(i)->dims->data[0]);
        for(int j = 1; j < interpreter->input_tensor(i)->dims->size; j++){
            input_dims_str += std::string("x");
            input_dims_str += std::to_string(interpreter->input_tensor(i)->dims->data[j]);
        }
        input_dims_str += std::string("]");

        std::cout << "INFO: Graph input " << i << ": " << interpreter->GetInputName(i) << " (" << TfLiteTypeGetName(interpreter->input_tensor(i)->type) << input_dims_str << ")" << std::endl;
    }

    // Print output tensors
    for(int i = 0; i < interpreter->outputs().size(); i++){
        std::string output_dims_str("[");
        output_dims_str += std::to_string(interpreter->output_tensor(i)->dims->data[0]);
        for(int j = 1; j < interpreter->output_tensor(i)->dims->size; j++){
            output_dims_str += std::string("x");
            output_dims_str += std::to_string(interpreter->output_tensor(i)->dims->data[j]);
        }
        output_dims_str += std::string("]");

        std::cout << "INFO: Graph output " << i << ": " << interpreter->GetOutputName(i) << " (" << TfLiteTypeGetName(interpreter->output_tensor(i)->type) << output_dims_str << ")" << std::endl;
    }
    
    // Call the appropriate functons
    if(strcmp(argv[1], "camera") == 0){
        // Parse the label
        std::ifstream labelfile(argv[4]);
        if(!labelfile.is_open()){
            std::cerr << "ERROR: Cannot open the label file.\n";
            return 1;
        }

        while(true){
            std::string line_string;

            if(std::getline(labelfile, line_string))
                labels.push_back(line_string);
            else
                break;
        }

        labelfile.close();

        std::cout << "INFO: Running the object detection using qcarcam API.\n";
        return run_qcarcam(interpreter.get(), model_mode, &labels, argv[5]);
    }
    else if(strcmp(argv[1], "image") == 0){
        // Parse the label
        std::ifstream labelfile(argv[4]);
        if(!labelfile.is_open()){
            std::cerr << "ERROR: Cannot open the label file.\n";
            return 1;
        }

        while(true){
            std::string line_string;

            if(std::getline(labelfile, line_string))
                labels.push_back(line_string);
            else
                break;
        }

        labelfile.close();

        // Parse batch size (optional)
        int batch_size = 1;
        if(argc > 7){
            batch_size = atoi(argv[7]);
        }
        
        // Parse score thresholds (optional)
        if(argc > 8){
            std::vector<float> ori_score_thrs;
            std::vector<float> new_score_thrs;
            std::string scores_arg(argv[8]);
            std::stringstream ss(scores_arg);
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    ori_score_thrs.push_back(std::stof(token));
                    if (std::getline(ss, token, ',')) {
                        new_score_thrs.push_back(std::stof(token));
                    }
                } catch (const std::invalid_argument& e) {
                    std::cerr << "ERROR: Invalid number format in score threshold arguments: " << token << std::endl;
                    return 1;
                }
            }

            interpreter->SetDynamicScoreThreshold(ori_score_thrs, new_score_thrs);
        }

        std::cout << "INFO: Running the object detection with jpeg images.\n";
        return run_image(interpreter.get(), model_mode, &labels, argv[5], argv[6], batch_size);
    }
    else if(strcmp(argv[1], "video") == 0){
        // Parse the label
        std::ifstream labelfile(argv[4]);
        if(!labelfile.is_open()){
            std::cerr << "ERROR: Cannot open the label file.\n";
            return 1;
        }

        while(true){
            std::string line_string;

            if(std::getline(labelfile, line_string))
                labels.push_back(line_string);
            else
                break;
        }

        labelfile.close();

        // Parse batch size (optional)
        int batch_size = 1;
        if(argc > 6){
            batch_size = atoi(argv[6]);
        }

        std::cout << "INFO: Running the object detection with a video.\n";
        return run_video(interpreter.get(), model_mode, &labels, argv[5], batch_size);
    }
    else if(strcmp(argv[1], "demo") == 0){
        // Parse the label
        std::ifstream labelfile(argv[4]);
        if(!labelfile.is_open()){
            std::cerr << "ERROR: Cannot open the label file.\n";
            return 1;
        }

        while(true){
            std::string line_string;

            if(std::getline(labelfile, line_string))
                labels.push_back(line_string);
            else
                break;
        }

        labelfile.close();

        // For demo mode, we use the all mode as the second interpreter for comparsion.
        std::string original_model_path(argv[2]);
        std::string model2_path;
        size_t last_dot_idx = original_model_path.find_last_of(".");
        if (last_dot_idx != std::string::npos) {
            model2_path = original_model_path.substr(0, last_dot_idx) + ".all";
        } else {
            model2_path = original_model_path + ".all";
        }

        std::unique_ptr<tflite::FlatBufferModel> model2 = tflite::FlatBufferModel::BuildFromFile(model2_path.c_str());
        if(model2 == NULL){
            std::cerr << "ERROR: Model2 load failed. Check the model2 name.\n";
            return 1;
        }

        tflite::ops::builtin::BuiltinOpResolver resolver2;
        tflite::InterpreterBuilder builder2(*model2, resolver2);
        std::unique_ptr<tflite::Interpreter> interpreter2;
        builder2(&interpreter2);
        if(interpreter2 == NULL){
            std::cerr << "ERROR: Interpreter build failed.\n";
            return 1;
        }

        if(interpreter2->AllocateTensors() != kTfLiteOk) {
            std::cerr << "ERROR: Memory allocation for interpreter2 failed.\n";
            return 1;
        }

        // Set number of threads for each accelerator
        #if defined(__aarch64__)
            interpreter2->SetNumThreads(1);
        #elif defined(__x86_64__)
            interpreter2->SetNumThreads(4);
        #endif

        // Print input tensors
        for(int i = 0; i < interpreter2->inputs().size(); i++){
            std::string input_dims_str("[");
            input_dims_str += std::to_string(interpreter2->input_tensor(i)->dims->data[0]);
            for(int j = 1; j < interpreter2->input_tensor(i)->dims->size; j++){
                input_dims_str += std::string("x");
                input_dims_str += std::to_string(interpreter2->input_tensor(i)->dims->data[j]);
            }
            input_dims_str += std::string("]");

            std::cout << "INFO: Graph input " << i << ": " << interpreter2->GetInputName(i) << " (" << TfLiteTypeGetName(interpreter2->input_tensor(i)->type) << input_dims_str << ")" << std::endl;
        }

        // Print output tensors
        for(int i = 0; i < interpreter2->outputs().size(); i++){
            std::string output_dims_str("[");
            output_dims_str += std::to_string(interpreter2->output_tensor(i)->dims->data[0]);
            for(int j = 1; j < interpreter2->output_tensor(i)->dims->size; j++){
                output_dims_str += std::string("x");
                output_dims_str += std::to_string(interpreter2->output_tensor(i)->dims->data[j]);
            }
            output_dims_str += std::string("]");

            std::cout << "INFO: Graph output " << i << ": " << interpreter2->GetOutputName(i) << " (" << TfLiteTypeGetName(interpreter2->output_tensor(i)->type) << output_dims_str << ")" << std::endl;
        }

        // Parse batch size (optional)
        int batch_size = 1;
        if(argc > 6){
            batch_size = atoi(argv[6]);
        }

        std::cout << "INFO: Running the demo mode.\n";
        return run_demo(interpreter.get(), interpreter2.get(), model_mode, &labels, argv[5], batch_size);
    }
    else if(strcmp(argv[1], "multicam") == 0){
        // Parse fusion mode
        std::string fusion_mode(argv[5]);
        if(fusion_mode != "sequential" && fusion_mode != "simultaneous") {
            std::cerr << "ERROR: Fusion mode must be 'sequential' or 'simultaneous'\n";
            return 1;
        }

        // Parse exec_times (optional)
        std::vector<int> exec_times;
        if (argc > 7) {
            std::string exec_times_arg(argv[7]);
            std::stringstream ss(exec_times_arg);
            std::string token;
            
            while (std::getline(ss, token, ',')) {
                size_t start = token.find_first_not_of(" \t\r\n");
                if (start == std::string::npos) {
                    continue;
                }
                size_t end = token.find_last_not_of(" \t\r\n");
                
                token = token.substr(start, end - start + 1);
                if (token.empty()) continue; 

                try {
                    size_t pos;
                    int val = std::stoi(token, &pos);
                    
                    if (pos != token.length()) {
                        std::cerr << "ERROR: Invalid characters trailing in exec_times argument: '" << token << "'\n";
                        return 1;
                    } 
                    
                    if (val < 0) {
                        std::cerr << "ERROR: Expected positive natural number, but got: '" << val << "'\n";
                        return 1;
                    }
                    
                    exec_times.push_back(val);
                    
                } catch (const std::invalid_argument& e) {
                    std::cerr << "ERROR: Invalid number format in exec_times argument: '" << token << "'\n";
                    return 1;
                } catch (const std::out_of_range& e) {
                    std::cerr << "ERROR: Number out of range in exec_times argument: '" << token << "'\n";
                    return 1;
                }
            }
            
            if (exec_times.size() != 4) {
                std::cerr << "ERROR: exec_times must contain exactly 4 values. Found " << exec_times.size() << ".\n";
                return 1;
            }
        }
        
        std::cout << "INFO: Running Multi-Camera Fusion\n";
        std::cout << "INFO: Target Fusion Mode: " << fusion_mode << "\n";
        
        if (!exec_times.empty()) {
            std::cout << "INFO: Optional exec_times:";
            for(size_t i = 0; i < exec_times.size(); i++){
                std::cout << " " << exec_times[i];
            }
            std::cout << "\n";
        }
        
        return run_multicam(interpreter.get(), model_mode, argv[4], argv[5], argv[6], exec_times);
    }

    return 0;
}
