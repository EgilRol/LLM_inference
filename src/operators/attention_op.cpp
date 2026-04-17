#include "operators/attention_op.h"

#include "kernel/kernels.cuh"

#include <cmath>

namespace {

void validate_attention_tensor(const runtime::DeviceTensorView<const float>& tensor,
                               const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("AttentionOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("AttentionOp: " + name + " must be rank-2");
}

void validate_attention_tensor(const runtime::DeviceTensorView<float>& tensor,
                               const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("AttentionOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("AttentionOp: " + name + " must be rank-2");
}

}  // namespace

AttentionOp::AttentionOp(size_t num_q_heads, size_t num_kv_heads, size_t head_dim)
    : num_q_heads_(num_q_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      scale_(1.0f / std::sqrt(static_cast<float>(head_dim))) {
  if (num_q_heads_ == 0 || num_kv_heads_ == 0)
    throw runtime_error("AttentionOp: head counts must be positive");
  if (num_q_heads_ % num_kv_heads_ != 0)
    throw runtime_error("AttentionOp: num_q_heads must be divisible by num_kv_heads");
  if (head_dim_ == 0)
    throw runtime_error("AttentionOp: head_dim must be positive");
}

void AttentionOp::forward(const runtime::CudaContext& context, runtime::Workspace& workspace,
                          runtime::DeviceTensorView<const float> q,
                          runtime::DeviceTensorView<const float> k,
                          runtime::DeviceTensorView<const float> v,
                          runtime::DeviceTensorView<float> output,
                          size_t q_position_offset, size_t k_position_offset) const {
  validate_attention_tensor(q, "q");
  validate_attention_tensor(k, "k");
  validate_attention_tensor(v, "v");
  validate_attention_tensor(output, "output");

  if (q.shape[1] != num_q_heads_ * head_dim_)
    throw runtime_error("AttentionOp: Q width does not match num_q_heads * head_dim");
  if (k.shape[1] != num_kv_heads_ * head_dim_)
    throw runtime_error("AttentionOp: K width does not match num_kv_heads * head_dim");
  if (v.shape != k.shape)
    throw runtime_error("AttentionOp: V shape must match K shape");
  if (output.shape[0] != q.shape[0] || output.shape[1] != num_q_heads_ * head_dim_)
    throw runtime_error("AttentionOp: output shape mismatch");

  runtime::DeviceTensorView<float> scores =
      workspace.get_view("attention_scores", {num_q_heads_ * q.shape[0], k.shape[0]});

  launch_attention_scores(context, q, k, scores, num_q_heads_, num_kv_heads_, head_dim_,
                          q_position_offset, k_position_offset, scale_);
  launch_softmax(context, scores);
  launch_attention_weighted_sum(
      context, runtime::DeviceTensorView<const float>(scores.data, scores.shape), v, output,
      num_q_heads_, num_kv_heads_, head_dim_);
}

size_t AttentionOp::num_q_heads() const { return num_q_heads_; }

size_t AttentionOp::num_kv_heads() const { return num_kv_heads_; }

size_t AttentionOp::head_dim() const { return head_dim_; }

float AttentionOp::scale() const { return scale_; }
