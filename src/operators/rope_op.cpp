#include "operators/rope_op.h"

#include "kernel/kernels.cuh"

namespace {

void validate_rope_tensor(const runtime::DeviceTensorView<float>& tensor, const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("RoPEOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("RoPEOp: " + name + " must be rank-2");
}

}  // namespace

RoPEOp::RoPEOp(runtime::DeviceTensorView<const float> cos_table,
               runtime::DeviceTensorView<const float> sin_table, size_t num_q_heads,
               size_t num_kv_heads, size_t head_dim)
    : cos_table_(std::move(cos_table)),
      sin_table_(std::move(sin_table)),
      num_q_heads_(num_q_heads),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim) {
  if (cos_table_.data == nullptr || sin_table_.data == nullptr)
    throw runtime_error("RoPEOp: cos/sin tables must have valid data pointers");
  if (cos_table_.shape.size() != 2 || sin_table_.shape.size() != 2)
    throw runtime_error("RoPEOp: cos/sin tables must be rank-2");
  if (cos_table_.shape != sin_table_.shape)
    throw runtime_error("RoPEOp: cos and sin table shapes must match");
  if (head_dim_ == 0 || head_dim_ % 2 != 0)
    throw runtime_error("RoPEOp: head_dim must be positive and even");
  if (cos_table_.shape[1] != head_dim_ / 2)
    throw runtime_error("RoPEOp: table width must equal head_dim / 2");
}

void RoPEOp::forward(const runtime::CudaContext& context, runtime::DeviceTensorView<float> q,
                     runtime::DeviceTensorView<float> k, size_t position_offset) const {
  validate_rope_tensor(q, "q");
  validate_rope_tensor(k, "k");

  if (q.shape[1] != num_q_heads_ * head_dim_)
    throw runtime_error("RoPEOp: Q width does not match num_q_heads * head_dim");
  if (k.shape[1] != num_kv_heads_ * head_dim_)
    throw runtime_error("RoPEOp: K width does not match num_kv_heads * head_dim");
  if (position_offset + q.shape[0] > cos_table_.shape[0] ||
      position_offset + k.shape[0] > cos_table_.shape[0]) {
    throw runtime_error("RoPEOp: requested positions exceed precomputed table");
  }

  launch_rope(context, q, cos_table_, sin_table_, num_q_heads_, head_dim_, position_offset);
  launch_rope(context, k, cos_table_, sin_table_, num_kv_heads_, head_dim_, position_offset);
}

runtime::DeviceTensorView<const float> RoPEOp::cos_table() const { return cos_table_; }

runtime::DeviceTensorView<const float> RoPEOp::sin_table() const { return sin_table_; }

size_t RoPEOp::num_q_heads() const { return num_q_heads_; }

size_t RoPEOp::num_kv_heads() const { return num_kv_heads_; }

size_t RoPEOp::head_dim() const { return head_dim_; }
