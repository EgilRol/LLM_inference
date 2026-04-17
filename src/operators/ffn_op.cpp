#include "operators/ffn_op.h"

namespace {

void validate_ffn_weight(const runtime::DeviceTensorView<const __nv_bfloat16>& weight,
                         const string& name) {
  if (weight.data == nullptr)
    throw runtime_error("FFNOp: " + name + " data pointer is null");
  if (weight.shape.size() != 2)
    throw runtime_error("FFNOp: " + name + " must be rank-2");
}

}  // namespace

FFNOp::FFNOp(runtime::DeviceTensorView<const __nv_bfloat16> gate_weight,
             runtime::DeviceTensorView<const __nv_bfloat16> up_weight,
             runtime::DeviceTensorView<const __nv_bfloat16> down_weight)
    : gate_weight_(std::move(gate_weight)),
      up_weight_(std::move(up_weight)),
      down_weight_(std::move(down_weight)),
      gate_proj_(gate_weight_),
      up_proj_(up_weight_),
      down_proj_(down_weight_) {
  validate_ffn_weight(gate_weight_, "gate_weight");
  validate_ffn_weight(up_weight_, "up_weight");
  validate_ffn_weight(down_weight_, "down_weight");

  if (gate_weight_.shape[1] != up_weight_.shape[1] ||
      gate_weight_.shape[1] != down_weight_.shape[0]) {
    throw runtime_error("FFNOp: incompatible FFN input/output dimensions");
  }
  if (gate_weight_.shape[0] != up_weight_.shape[0]) {
    throw runtime_error("FFNOp: gate and up projections must share the same hidden dimension");
  }
  if (down_weight_.shape[1] != gate_weight_.shape[0]) {
    throw runtime_error("FFNOp: down projection input dim must match SwiGLU hidden dim");
  }
}

void FFNOp::forward(const runtime::CudaContext& context, runtime::Workspace& workspace,
                    runtime::DeviceTensorView<const float> input,
                    runtime::DeviceTensorView<float> output) const {
  if (input.data == nullptr)
    throw runtime_error("FFNOp: input data pointer is null");
  if (output.data == nullptr)
    throw runtime_error("FFNOp: output data pointer is null");
  if (input.shape.size() != 2 || output.shape.size() != 2)
    throw runtime_error("FFNOp: input and output must be rank-2");
  if (input.shape[1] != in_dim())
    throw runtime_error("FFNOp: input hidden dimension does not match FFN weights");
  if (output.shape[0] != input.shape[0] || output.shape[1] != out_dim())
    throw runtime_error("FFNOp: output shape does not match FFN output dimension");

  runtime::DeviceTensorView<float> gate =
      workspace.get_view("ffn_gate", {input.shape[0], hidden_dim()});
  runtime::DeviceTensorView<float> up =
      workspace.get_view("ffn_up", {input.shape[0], hidden_dim()});
  runtime::DeviceTensorView<float> fused =
      workspace.get_view("ffn_swiglu", {input.shape[0], hidden_dim()});

  gate_proj_.forward(context, input, gate);
  up_proj_.forward(context, input, up);
  swiglu_.forward(context, runtime::DeviceTensorView<const float>(gate.data, gate.shape),
                  runtime::DeviceTensorView<const float>(up.data, up.shape), fused);
  down_proj_.forward(context, runtime::DeviceTensorView<const float>(fused.data, fused.shape),
                     output);
}

size_t FFNOp::in_dim() const { return gate_weight_.shape[1]; }

size_t FFNOp::hidden_dim() const { return gate_weight_.shape[0]; }

size_t FFNOp::out_dim() const { return down_weight_.shape[0]; }
