#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_SERIALIZATION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_SERIALIZATION_H_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tflite/delegates/gpu/common/status.h"
#include "tflite/delegates/gpu/common/types.h"
#include "tflite/delegates/gpu/gl/compiled_model_generated.h"
#include "tflite/delegates/gpu/gl/object.h"
#include "tflite/delegates/gpu/gl/variable.h"

// [Fix] Include gl_shader.h to use BinaryShader struct.
#include "tflite/delegates/gpu/gl/gl_shader.h"

namespace tflite {
namespace gpu {
namespace gl {

struct CompiledModelOptions {
  // If true, a model was compiled with dynamic batch size and therefore,
  // a user may change BATCH dimension at runtime.
  bool dynamic_batch = false;
};

// Accumulates shaders and programs and stores it in FlatBuffer format.
class SerializedCompiledModelBuilder {
 public:
  SerializedCompiledModelBuilder() : builder_(32 * 1024) {}

  // [Fix] Modified to take the compiled BinaryShader instead of a source string.
  void AddShader(const BinaryShader& binary_shader);

  void AddProgram(const std::vector<Variable>& parameters,
                  const std::vector<Object>& objects,
                  const uint3& workgroup_size, const uint3& num_workgroups,
                  size_t shader_index);

  // Returns serialized data that will stay valid until this object is
  // destroyed.
  absl::Span<const uint8_t> Finalize(const CompiledModelOptions& options);

 private:
  std::vector<flatbuffers::Offset<flatbuffers::String>> shaders_;
  std::vector<flatbuffers::Offset<data::Program>> programs_;
  ::flatbuffers::FlatBufferBuilder builder_;
};

// Handles deserialization events. it is guaranteed that shaders will be called
// first in the appropriate order and programs come next.
class DeserializationHandler {
 public:
  virtual ~DeserializationHandler() = default;

  // [Fix] Modified to provide the extracted BinaryShader to bypass compilation.
  virtual absl::Status OnShader(const BinaryShader& binary_shader) = 0;

  virtual absl::Status OnProgram(const std::vector<Variable>& parameters,
                                 const std::vector<Object>& objects,
                                 const uint3& workgroup_size,
                                 const uint3& num_workgroups,
                                 size_t shader_index) = 0;

  virtual void OnOptions(const CompiledModelOptions& options) = 0;
};

absl::Status DeserializeCompiledModel(absl::Span<const uint8_t> serialized,
                                      DeserializationHandler* handler);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_SERIALIZATION_H_
