#pragma once

#include "runtime/cuda_context.h"
#include "runtime/device_tensor.h"
#include "runtime/workspace.h"

class AttentionOp {
 public:
  AttentionOp(size_t num_q_heads, size_t num_kv_heads, size_t head_dim);

  // Runs causal grouped-query attention and writes [q_tokens, num_q_heads * head_dim].
  void forward(const runtime::CudaContext& context, runtime::Workspace& workspace,
               runtime::DeviceTensorView<const float> q,
               runtime::DeviceTensorView<const float> k,
               runtime::DeviceTensorView<const float> v,
               runtime::DeviceTensorView<float> output, size_t q_position_offset = 0,
               size_t k_position_offset = 0) const;

  size_t num_q_heads() const;
  size_t num_kv_heads() const;
  size_t head_dim() const;
  float scale() const;

 private:
  size_t num_q_heads_;
  size_t num_kv_heads_;
  size_t head_dim_;
  float scale_;
};
