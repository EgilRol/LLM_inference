#include "operators/qkv_projection_op.h"

namespace {

void validate_projection_weight(const runtime::DeviceTensorView<const __nv_bfloat16>& weight,
                                const string& name) {
  if (weight.data == nullptr)
    throw runtime_error("QKVProjectionOp: " + name + " data pointer is null");
  if (weight.shape.size() != 2)
    throw runtime_error("QKVProjectionOp: " + name + " must be rank-2");
}

}  // namespace

QKVProjectionOp::QKVProjectionOp(runtime::DeviceTensorView<const __nv_bfloat16> q_weight,
                                 runtime::DeviceTensorView<const __nv_bfloat16> k_weight,
                                 runtime::DeviceTensorView<const __nv_bfloat16> v_weight)
    : q_weight_(std::move(q_weight)),
      k_weight_(std::move(k_weight)),
      v_weight_(std::move(v_weight)),
      q_proj_(q_weight_),
      k_proj_(k_weight_),
      v_proj_(v_weight_) {
  validate_projection_weight(q_weight_, "q_weight");
  validate_projection_weight(k_weight_, "k_weight");
  validate_projection_weight(v_weight_, "v_weight");

  if (q_weight_.shape[1] != k_weight_.shape[1] || q_weight_.shape[1] != v_weight_.shape[1]) {
    throw runtime_error("QKVProjectionOp: Q, K, and V weights must share the same input dim");
  }
}

void QKVProjectionOp::forward(const runtime::CudaContext& context,
                              runtime::DeviceTensorView<const float> input,
                              runtime::DeviceTensorView<float> q,
                              runtime::DeviceTensorView<float> k,
                              runtime::DeviceTensorView<float> v) const {
  if (input.data == nullptr)
    throw runtime_error("QKVProjectionOp: input data pointer is null");
  if (q.data == nullptr || k.data == nullptr || v.data == nullptr)
    throw runtime_error("QKVProjectionOp: output data pointer is null");
  if (input.shape.size() != 2 || q.shape.size() != 2 || k.shape.size() != 2 || v.shape.size() != 2)
    throw runtime_error("QKVProjectionOp: input and outputs must be rank-2");
  if (input.shape[1] != in_dim())
    throw runtime_error("QKVProjectionOp: input hidden dimension does not match weights");
  if (q.shape[0] != input.shape[0] || q.shape[1] != q_dim())
    throw runtime_error("QKVProjectionOp: Q output shape mismatch");
  if (k.shape[0] != input.shape[0] || k.shape[1] != k_dim())
    throw runtime_error("QKVProjectionOp: K output shape mismatch");
  if (v.shape[0] != input.shape[0] || v.shape[1] != v_dim())
    throw runtime_error("QKVProjectionOp: V output shape mismatch");

  q_proj_.forward(context, input, q);
  k_proj_.forward(context, input, k);
  v_proj_.forward(context, input, v);
}

size_t QKVProjectionOp::in_dim() const { return q_weight_.shape[1]; }

size_t QKVProjectionOp::q_dim() const { return q_weight_.shape[0]; }

size_t QKVProjectionOp::k_dim() const { return k_weight_.shape[0]; }

size_t QKVProjectionOp::v_dim() const { return v_weight_.shape[0]; }
