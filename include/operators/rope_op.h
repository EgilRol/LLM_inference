#pragma once

#include "runtime/cuda_context.h"
#include "runtime/device_tensor.h"
#include "runtime/rope_tables.h"

class RoPEOp {
 public:
  // Applies RoPE in place to flat [tokens, heads * head_dim] Q/K tensors.
  RoPEOp(runtime::DeviceTensorView<const float> cos_table,
         runtime::DeviceTensorView<const float> sin_table, size_t num_q_heads,
         size_t num_kv_heads, size_t head_dim);

  void forward(const runtime::CudaContext& context, runtime::DeviceTensorView<float> q,
               runtime::DeviceTensorView<float> k, size_t position_offset = 0) const;

  runtime::DeviceTensorView<const float> cos_table() const;
  runtime::DeviceTensorView<const float> sin_table() const;
  size_t num_q_heads() const;
  size_t num_kv_heads() const;
  size_t head_dim() const;

 private:
  runtime::DeviceTensorView<const float> cos_table_;
  runtime::DeviceTensorView<const float> sin_table_;
  size_t num_q_heads_;
  size_t num_kv_heads_;
  size_t head_dim_;
};
