// Copyright ⓒ 2019- Mobilint Inc. All rights reserved.

#ifndef MACCEL_MODEL_H_
#define MACCEL_MODEL_H_

#include <cstdint>
#ifndef _MSC_VER
#include <experimental/propagate_const>
#endif
#include <memory>
#include <string>
#include <vector>

#include "maccel/export.h"
#include "maccel/ndarray.h"
#include "maccel/type.h"

namespace mobilint {
class Accelerator;
class ModelImpl;

class MACCEL_EXPORT Model {
public:
    static std::unique_ptr<Model> create(const std::string& mxq_path, StatusCode& sc);
    static std::unique_ptr<Model> create(const std::string& mxq_path,
                                         const ModelConfig& config, StatusCode& sc);
    Model(const Model& other) = delete;
    Model(Model&& other) noexcept;
    Model& operator=(const Model& rhs) = delete;
    Model& operator=(Model&& rhs) noexcept;
    ~Model();

    StatusCode launch(Accelerator& acc);
    StatusCode dispose();
    bool isTarget(CoreId core_id) const;

    std::vector<CoreId> getTargetCores() const;

    // 1. float -> float inference
    StatusCode infer(const std::vector<NDArray<float>>& input,
                     std::vector<NDArray<float>>& output);
    StatusCode inferCHW(const std::vector<NDArray<float>>& input,
                        std::vector<NDArray<float>>& output);
    std::vector<NDArray<float>> infer(const std::vector<NDArray<float>>& input,
                                      StatusCode& sc);
    std::vector<NDArray<float>> inferCHW(const std::vector<NDArray<float>>& input,
                                         StatusCode& sc);

    // 2. int8 -> int8 inferenece
    StatusCode infer(const std::vector<NDArray<int8_t>>& input,
                     std::vector<NDArray<int8_t>>& output);
    StatusCode inferCHW(const std::vector<NDArray<int8_t>>& input,
                        std::vector<NDArray<int8_t>>& output);
    std::vector<NDArray<int8_t>> infer(const std::vector<NDArray<int8_t>>& input,
                                       StatusCode& sc);
    std::vector<NDArray<int8_t>> inferCHW(const std::vector<NDArray<int8_t>>& input,
                                          StatusCode& sc);

    // 3. int8 -> float inference
    std::vector<NDArray<float>> inferToFloat(const std::vector<NDArray<int8_t>>& input,
                                             StatusCode& sc);
    std::vector<NDArray<float>> inferCHWToFloat(const std::vector<NDArray<int8_t>>& input,
                                                StatusCode& sc);

    // 4. Etc.
    // NPU-buffer inference.
    StatusCode inferBuffer(const std::vector<Buffer>& input, std::vector<Buffer>& output);
    // NPU-buffer inference (batch version): [num_inputs, batch] -> [num_outputs, batch].
    StatusCode inferBuffer(const std::vector<std::vector<Buffer>>& input,
                           std::vector<std::vector<Buffer>>& output);
    StatusCode inferBufferToFloat(const std::vector<Buffer>& input,
                                  std::vector<std::vector<float>>& output);
    StatusCode inferBufferToFloat(const std::vector<std::vector<Buffer>>& input,
                                  std::vector<std::vector<float>>& output);
    StatusCode inferBufferToFloat(const std::vector<Buffer>& input,
                                  std::vector<NDArray<float>>& output);
    StatusCode inferBufferToFloat(const std::vector<std::vector<Buffer>>& input,
                                  std::vector<NDArray<float>>& output);
    // infer for speed test.
    StatusCode inferSpeedrun();

    // 5. `std::vector`-based APIs (advised not to use.)
    // 5.1. float -> float inference
    StatusCode infer(const std::vector<float*>& input,
                     std::vector<std::vector<float>>& output);
    StatusCode inferCHW(const std::vector<float*>& input,
                        std::vector<std::vector<float>>& output);
    std::vector<std::vector<float>> infer(const std::vector<float*>& input,
                                          StatusCode& sc);
    std::vector<std::vector<float>> inferCHW(const std::vector<float*>& input,
                                             StatusCode& sc);
    // NOTE: Experimental: infer with height batch.
    StatusCode inferHeightBatch(const std::vector<float*>& input,
                                std::vector<std::vector<float>>& output,
                                int height_batch_size);
    // NOTE: Experimental: inference API using multiple batch.
    // [num_inputs, batch * W * H * C] -> [num_outputs, batch * H * W * C].
    StatusCode infer(const std::vector<float*>& input,
                     std::vector<std::vector<float>>& output, int batch_size);
    std::vector<std::vector<float>> infer(const std::vector<float*>& input,
                                          int batch_size, StatusCode& sc);

    // 5.2. int8 -> int8 inferenece
    StatusCode infer(const std::vector<int8_t*>& input,
                     std::vector<std::vector<int8_t>>& output);
    StatusCode inferCHW(const std::vector<int8_t*>& input,
                        std::vector<std::vector<int8_t>>& output);
    std::vector<std::vector<int8_t>> infer(const std::vector<int8_t*>& input,
                                           StatusCode& sc);
    std::vector<std::vector<int8_t>> inferCHW(const std::vector<int8_t*>& input,
                                              StatusCode& sc);

    // 5.3. int8 -> float inference
    std::vector<std::vector<float>> inferToFloat(const std::vector<int8_t*>& input,
                                                 StatusCode& sc);
    std::vector<std::vector<float>> inferCHWToFloat(const std::vector<int8_t*>& input,
                                                    StatusCode& sc);

    // Reposition single batch.
    StatusCode repositionInputs(const std::vector<float*>& input,
                                std::vector<Buffer>& input_buf) const;
    StatusCode repositionOutputs(const std::vector<Buffer>& output_buf,
                                 std::vector<float*>& output) const;
    StatusCode repositionOutputs(const std::vector<Buffer>& output_buf,
                                 std::vector<std::vector<float>>& output) const;
    // Reposition multiple batches.
    StatusCode repositionInputs(const std::vector<float*>& input,
                                std::vector<std::vector<Buffer>>& input_buf) const;
    StatusCode repositionOutputs(const std::vector<std::vector<Buffer>>& output_buf,
                                 std::vector<float*>& output) const;
    StatusCode repositionOutputs(const std::vector<std::vector<Buffer>>& output_buf,
                                 std::vector<std::vector<float>>& output) const;
    // (experimental) infer using packed MXQs
    StatusCode repositionToPackedInputs(
        const std::vector<std::vector<float*>>& input,
        std::vector<std::vector<Buffer>>& input_buf) const;
    StatusCode inferPacked(const std::vector<std::vector<Buffer>>& input_buf,
                           std::vector<std::vector<Buffer>>& output_buf);
    StatusCode repositionFromPackedOutputs(
        const std::vector<std::vector<Buffer>>& output_buf,
        std::vector<std::vector<float*>>& output) const;

    const std::vector<std::vector<int64_t>>& getModelInputShape() const;
    const std::vector<std::vector<int64_t>>& getModelOutputShape() const;

    std::vector<Scale> getInputScale() const;
    std::vector<Scale> getOutputScale() const;
    const std::vector<BufferInfo>& getInputBufferInfo() const;
    const std::vector<BufferInfo>& getOutputBufferInfo() const;
    std::vector<Buffer> acquireInputBuffer() const;
    std::vector<Buffer> acquireOutputBuffer() const;
    std::vector<std::vector<Buffer>> acquireInputBuffers(const int batch_size) const;
    std::vector<std::vector<Buffer>> acquireOutputBuffers(const int batch_size) const;
    StatusCode releaseBuffer(std::vector<Buffer>& buffer) const;
    StatusCode releaseBuffers(std::vector<std::vector<Buffer>>& buffers) const;

    std::vector<std::vector<Buffer>> acquirePackedInputBuffer() const;
    std::vector<std::vector<Buffer>> acquirePackedOutputBuffer() const;
    StatusCode releasePackedBuffer(std::vector<std::vector<Buffer>>& buffer) const;

    uint32_t getIdentifier() const;
    std::string getModelPath() const;
    SchedulePolicy getSchedulePolicy() const;
    LatencySetPolicy getLatencySetPolicy() const;
    MaintenancePolicy getMaintenancePolicy() const;
    uint64_t getLatencyConsumed(const int npu_op_idx) const;
    uint64_t getLatencyFinished(const int npu_op_idx) const;
    std::shared_ptr<Statistics> getStatistics() const;

private:
    Model();

#ifndef _MSC_VER
    std::experimental::propagate_const<std::unique_ptr<ModelImpl>> mImpl;
#else
    std::unique_ptr<ModelImpl> mImpl;
#endif

    friend class Accelerator;
};
}  // namespace mobilint

#endif
