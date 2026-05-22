#ifndef _ENGINE_INTERFACE_
#define _ENGINE_INTERFACE_

#include <memory>
#include <vector>
#include <string>

#if __has_include("tflite/interpreter.h")
    #include <tflite/interpreter.h>
    #include <tflite/kernels/register.h>
    #include <tflite/model.h>
    #include <tflite/delegates/hexagon/hexagon_delegate.h>
    #include <tflite/delegates/gpu/delegate.h>
#elif __has_include("tensorflow/lite/interpreter.h")
    #include <tensorflow/lite/interpreter.h>
    #include <tensorflow/lite/kernels/register.h>
    #include <tensorflow/lite/model.h>
    #include <tensorflow/lite/delegates/hexagon/hexagon_delegate.h>
    #include <tensorflow/lite/delegates/gpu/delegate.h>
#else
    #error "Cannot find tflite or tensorflow/lite header"
#endif

#if __has_include("qbruntime/qbruntime.h")
    #include <qbruntime/qbruntime.h>
#elif __has_include("maccel/maccel.h")
    #include <maccel/maccel.h>
#else
    #error "Cannot find qbruntime or maccel header"
#endif

#include <hailo/hailort.hpp>

namespace tflite {
    namespace pkshin {
        class Interpreter; // Forward declaration

        TfLiteDelegate* TfLiteGpuDelegateV2Create(const TfLiteGpuDelegateOptionsV2* options);
        void TfLiteGpuDelegateV2Delete(TfLiteDelegate* delegate);

        TfLiteDelegate* TfLiteHexagonDelegateCreate(const TfLiteHexagonDelegateOptions* options);
        void TfLiteHexagonDelegateDelete(TfLiteDelegate* delegate);
        void TfLiteHexagonInit();
        void TfLiteHexagonInitWithPath(const char* lib_directory_path);
        void TfLiteHexagonTearDown();

        class FlatBufferModel : public ::tflite::FlatBufferModel {
        public:
            static std::unique_ptr<FlatBufferModel> BuildFromFile(
                const char* filename, ::tflite::ErrorReporter* error_reporter = ::tflite::DefaultErrorReporter());
            FlatBufferModel();
            ~FlatBufferModel();
        private:
            friend class InterpreterBuilder;
        };

        class InterpreterBuilder : public ::tflite::InterpreterBuilder {
        public:
            InterpreterBuilder(const FlatBufferModel& model, const ::tflite::OpResolver& op_resolver);
            ~InterpreterBuilder();
            TfLiteStatus operator()(std::unique_ptr<Interpreter>* interpreter);
        };

        class Interpreter : public ::tflite::Interpreter {
        public:
            ~Interpreter();
            TfLiteStatus Invoke();
            TfLiteStatus ResizeInputTensor(int tensor_index, const std::vector<int>& dims);
            TfLiteStatus SetNumThreads(int num_threads);
            TfLiteStatus ModifyGraphWithDelegate(TfLiteDelegate* delegate);
            TfLiteStatus AllocateTensors();

            const std::vector<int>& inputs() const;
            const std::vector<int>& outputs() const;
            
            TfLiteTensor* tensor(int tensor_index);
            const TfLiteTensor* tensor(int tensor_index) const;
            TfLiteTensor* input_tensor(int index);
            const TfLiteTensor* input_tensor(int index) const;
            TfLiteTensor* output_tensor(int index);
            const TfLiteTensor* output_tensor(int index) const;
            
            const char* GetInputName(int index) const;
            const char* GetOutputName(int index) const;
            
            // --- Custom public methods ---
            const std::vector<int>& tflite_inputs() const;
            const std::vector<int>& tflite_outputs() const;
            const std::vector<int>& hailo_inputs() const;
            const std::vector<int>& hailo_outputs() const;
            const std::vector<int>& maccel_inputs() const;
            const std::vector<int>& maccel_outputs() const;

            TfLiteTensor* tflite_input_tensor(int index);
            const TfLiteTensor* tflite_input_tensor(int index) const;
            TfLiteTensor* tflite_output_tensor(int index);
            const TfLiteTensor* tflite_output_tensor(int index) const;

            TfLiteTensor* maccel_input_tensor(int index);
            const TfLiteTensor* maccel_input_tensor(int index) const;
            TfLiteTensor* maccel_output_tensor(int index);
            const TfLiteTensor* maccel_output_tensor(int index) const;

            TfLiteTensor* hailo_input_tensor(int index);
            const TfLiteTensor* hailo_input_tensor(int index) const;
            TfLiteTensor* hailo_output_tensor(int index);
            const TfLiteTensor* hailo_output_tensor(int index) const;

            void* get_input_data_ptr(int index, int batch_index = 0);
            const void* get_input_data_ptr(int index, int batch_index = 0) const;
            void* get_output_data_ptr(int index, int batch_index = 0);
            const void* get_output_data_ptr(int index, int batch_index = 0) const;

            template <class T>
            T* typed_input_tensor(int index, int batch_index = 0) {
                return static_cast<T*>(get_input_data_ptr(index, batch_index));
            }
            template <class T>
            const T* typed_input_tensor(int index, int batch_index = 0) const {
                return static_cast<const T*>(get_input_data_ptr(index, batch_index));
            }
            
            template <class T>
            T* typed_output_tensor(int index, int batch_index = 0) {
                return static_cast<T*>(get_output_data_ptr(index, batch_index));
            }
            template <class T>
            const T* typed_output_tensor(int index, int batch_index = 0) const {
                return static_cast<const T*>(get_output_data_ptr(index, batch_index));
            }

            template <class T> T* typed_tflite_input_tensor(int index, int batch_index = 0);
            template <class T> const T* typed_tflite_input_tensor(int index, int batch_index = 0) const;
            template <class T> T* typed_tflite_output_tensor(int index, int batch_index = 0);
            template <class T> const T* typed_tflite_output_tensor(int index, int batch_index = 0) const;
            
            template <class T> T* typed_maccel_input_tensor(int index, int batch_index = 0);
            template <class T> const T* typed_maccel_input_tensor(int index, int batch_index = 0) const;
            template <class T> T* typed_maccel_output_tensor(int index, int batch_index = 0);
            template <class T> const T* typed_maccel_output_tensor(int index, int batch_index = 0) const;
            
            template <class T> T* typed_hailo_input_tensor(int index, int batch_index = 0);
            template <class T> const T* typed_hailo_input_tensor(int index, int batch_index = 0) const;
            template <class T> T* typed_hailo_output_tensor(int index, int batch_index = 0);
            template <class T> const T* typed_hailo_output_tensor(int index, int batch_index = 0) const;

            TfLiteStatus SetAcceleratorMask(const std::vector<bool>& mask);
            std::vector<bool> GetAcceleratorMask() const;

            TfLiteStatus Invoke(bool async, const std::vector<int>& execution_times_ms = {}, int deadline_ms = 0);

            void WaitForCompletion(int batch_index = -1);

            bool is_tflite_model() const;
            bool is_hailo_output(int batch_id = 0) const;
            bool is_maccel_output(int batch_id = 0) const;
            bool is_tflite_output(int batch_id = 0) const;

            TfLiteStatus SetDynamicScoreThreshold(std::vector<float> ori_score_thrs, std::vector<float> new_score_thrs);
            float GetHailoScoreThreshold() const;
            float GetMaccelScoreThreshold() const;

            TfLiteStatus SetScoreThreshold(float score_thrs);
            TfLiteStatus SetIouThreshold(float iou_thrs);
            
            double GetSumTurnAroundTime() const;
            double GetMaxTurnAroundTime() const;

            double GetTheoreticalMaxTime() const;
        private:
            friend class InterpreterBuilder;
            Interpreter();
        };

    } // namespace pkshin
} // namespace tflite

namespace tflite_proxy {
    using namespace ::tflite;

    using ::tflite::pkshin::FlatBufferModel;
    using ::tflite::pkshin::Interpreter;
    using ::tflite::pkshin::InterpreterBuilder;
}

#define tflite tflite_proxy

#define TfLiteGpuDelegateV2Create ::tflite::pkshin::TfLiteGpuDelegateV2Create
#define TfLiteGpuDelegateV2Delete ::tflite::pkshin::TfLiteGpuDelegateV2Delete
#define TfLiteHexagonDelegateCreate ::tflite::pkshin::TfLiteHexagonDelegateCreate
#define TfLiteHexagonDelegateDelete ::tflite::pkshin::TfLiteHexagonDelegateDelete
#define TfLiteHexagonInit ::tflite::pkshin::TfLiteHexagonInit
#define TfLiteHexagonInitWithPath ::tflite::pkshin::TfLiteHexagonInitWithPath
#define TfLiteHexagonTearDown ::tflite::pkshin::TfLiteHexagonTearDown

#endif //_ENGINE_INTERFACE_