#pragma once

#include "io/staged_reader.h"
#include "runtime/cuda_context.h"
#include "runtime/device_buffer.h"

namespace runtime {

struct OwnedTensorGPU {
  io::TensorMeta meta;
  // Persistent GPU allocation for one model weight tensor.
  DeviceBuffer<__nv_bfloat16> buffer;

  DeviceTensorView<__nv_bfloat16> view();
  DeviceTensorView<const __nv_bfloat16> view() const;
};

struct LayerWeightsGPU {
  OwnedTensorGPU input_layernorm;
  OwnedTensorGPU post_attention_layernorm;
  OwnedTensorGPU q_proj;
  OwnedTensorGPU k_proj;
  OwnedTensorGPU v_proj;
  OwnedTensorGPU o_proj;
  OwnedTensorGPU gate_proj;
  OwnedTensorGPU up_proj;
  OwnedTensorGPU down_proj;
};

class ModelWeightsGPU {
 public:
  // Loads the dumped model weights directly into their final GPU allocations.
  explicit ModelWeightsGPU(const string& weights_dir, io::StagedReader& staged_reader,
                           const CudaContext& context);

  static constexpr size_t kNumLayers = 32;

  const LayerWeightsGPU& layer(size_t layer_idx) const;
  DeviceTensorView<const __nv_bfloat16> embed_tokens() const;
  DeviceTensorView<const __nv_bfloat16> lm_head() const;
  DeviceTensorView<const __nv_bfloat16> model_norm() const;

 private:
  static OwnedTensorGPU load_owned_tensor(const string& file_path, const string& tensor_name,
                                          io::StagedReader& staged_reader,
                                          const CudaContext& context);
  static string layer_file_path(const string& weights_dir, size_t layer_idx);

  OwnedTensorGPU embed_tokens_;
  OwnedTensorGPU lm_head_;
  OwnedTensorGPU model_norm_;
  vector<LayerWeightsGPU> layers_;
};

}  // namespace runtime
