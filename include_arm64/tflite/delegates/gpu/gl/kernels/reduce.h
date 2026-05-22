#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_REDUCE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_REDUCE_H_

#include <memory>

#include "tflite/delegates/gpu/common/operations.h"
#include "tflite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {

// Creates a generic reduce shader based on the specific OperationType.
// Supported types: REDUCE_MAXIMUM, REDUCE_MINIMUM, REDUCE_PRODUCT, REDUCE_SUM.
std::unique_ptr<NodeShader> NewReduceNodeShader(OperationType op_type);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_REDUCE_H_