#include "runtime/model_weights_gpu.h"

#include <iomanip>
#include <sstream>

namespace runtime {
namespace {

string join_path(const string& dir, const string& file_name) {
  if (!dir.empty() && dir.back() == '/')
    return dir + file_name;
  return dir + "/" + file_name;
}

string layer_prefix(size_t layer_idx) {
  return "model.layers." + std::to_string(layer_idx) + ".";
}

}  // namespace

DeviceTensorView<__nv_bfloat16> OwnedTensorGPU::view() { return buffer.view(meta.shape); }

DeviceTensorView<const __nv_bfloat16> OwnedTensorGPU::view() const {
  return buffer.view(meta.shape);
}

ModelWeightsGPU::ModelWeightsGPU(const string& weights_dir, io::StagedReader& staged_reader,
                                 const CudaContext& context)
    : embed_tokens_(load_owned_tensor(join_path(weights_dir, "embed_tokens.bin"),
                                      "model.embed_tokens.weight", staged_reader, context)),
      lm_head_(load_owned_tensor(join_path(weights_dir, "lm_head.bin"), "lm_head.weight",
                                 staged_reader, context)),
      model_norm_(load_owned_tensor(join_path(weights_dir, "norm.bin"), "model.norm.weight",
                                    staged_reader, context)),
      layers_(kNumLayers) {
  for (size_t layer_idx = 0; layer_idx < kNumLayers; ++layer_idx) {
    const string file_path = layer_file_path(weights_dir, layer_idx);
    const string prefix = layer_prefix(layer_idx);
    LayerWeightsGPU layer;
    layer.input_layernorm = load_owned_tensor(
        file_path, prefix + "input_layernorm.weight", staged_reader, context);
    layer.post_attention_layernorm = load_owned_tensor(
        file_path, prefix + "post_attention_layernorm.weight", staged_reader, context);
    layer.q_proj =
        load_owned_tensor(file_path, prefix + "self_attn.q_proj.weight", staged_reader, context);
    layer.k_proj =
        load_owned_tensor(file_path, prefix + "self_attn.k_proj.weight", staged_reader, context);
    layer.v_proj =
        load_owned_tensor(file_path, prefix + "self_attn.v_proj.weight", staged_reader, context);
    layer.o_proj =
        load_owned_tensor(file_path, prefix + "self_attn.o_proj.weight", staged_reader, context);
    layer.gate_proj =
        load_owned_tensor(file_path, prefix + "mlp.gate_proj.weight", staged_reader, context);
    layer.up_proj =
        load_owned_tensor(file_path, prefix + "mlp.up_proj.weight", staged_reader, context);
    layer.down_proj =
        load_owned_tensor(file_path, prefix + "mlp.down_proj.weight", staged_reader, context);
    layers_[layer_idx] = std::move(layer);
  }
}

const LayerWeightsGPU& ModelWeightsGPU::layer(size_t layer_idx) const {
  if (layer_idx >= layers_.size())
    throw runtime_error("ModelWeightsGPU: layer index out of range");
  return layers_[layer_idx];
}

DeviceTensorView<const __nv_bfloat16> ModelWeightsGPU::embed_tokens() const {
  return embed_tokens_.view();
}

DeviceTensorView<const __nv_bfloat16> ModelWeightsGPU::lm_head() const {
  return lm_head_.view();
}

DeviceTensorView<const __nv_bfloat16> ModelWeightsGPU::model_norm() const {
  return model_norm_.view();
}

OwnedTensorGPU ModelWeightsGPU::load_owned_tensor(const string& file_path,
                                                  const string& tensor_name,
                                                  io::StagedReader& staged_reader,
                                                  const CudaContext& context) {
  io::WeightLoader loader(file_path);
  OwnedTensorGPU tensor;
  tensor.meta = loader.meta(tensor_name);
  if (tensor.meta.dtype != io::TensorDType::BF16) {
    throw runtime_error("ModelWeightsGPU: expected BF16 tensor '" + tensor_name + "'");
  }
  tensor.buffer.resize(tensor.meta.num_elements);
  // Startup does disk -> staging buffer -> final device allocation exactly once.
  staged_reader.upload_tensor(loader, tensor_name, tensor.buffer.data(), context.stream());
  context.synchronize();
  return tensor;
}

string ModelWeightsGPU::layer_file_path(const string& weights_dir, size_t layer_idx) {
  std::ostringstream name;
  name << "layer_" << std::setw(2) << std::setfill('0') << layer_idx << ".bin";
  return join_path(weights_dir, name.str());
}

}  // namespace runtime
