#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_SPLIT_H
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_SPLIT_H

#include <memory>

#include "tflite/delegates/gpu/common/operations.h"
#include "tflite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {

std::unique_ptr<NodeShader> NewSplitNodeShader();

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_KERNELS_SPLIT_H
