#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_GATHER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_GATHER_H_

#include <memory>

#include "tflite/delegates/gpu/common/operations.h"
#include "tflite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {

std::unique_ptr<NodeShader> NewGatherNodeShader();

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_GATHER_H_
