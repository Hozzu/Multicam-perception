#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_BATCHED_MATMUL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_BATCHED_MATMUL_H_

#include "tflite/delegates/gpu/common/operations.h"
#include "tflite/delegates/gpu/common/status.h"
#include "tflite/delegates/gpu/common/task/gpu_operation.h"
#include "tflite/delegates/gpu/common/gpu_info.h"

namespace tflite {
namespace gpu {

class BatchedMatMul : public GPUOperation {
 public:
  BatchedMatMul() = default;
  
  // =========================================================================
  // [CRITICAL FIX] Updated Constructor Signature
  // Added 'const GpuInfo& gpu_info' to allow dynamic SSBO allocation for constants
  // =========================================================================
  BatchedMatMul(const GpuInfo& gpu_info, 
                const OperationDef& definition,
                const BatchedMatMulAttributes& attr);
                
  int3 GetGridSize() const override;
};

// =========================================================================
// [CRITICAL FIX] Updated Factory Function Signature
// Matches the call from operation_selector.cc
// =========================================================================
BatchedMatMul CreateBatchedMatMul(const GpuInfo& gpu_info,
                                  const OperationDef& definition,
                                  const BatchedMatMulAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_BATCHED_MATMUL_H_