#pragma once

#include "operators/linear_op.h"
#include "runtime/cuda_context.h"
#include "runtime/device_tensor.h"

class QKVProjectionOp {
 public:
  // Binds checkpoint-layout Q/K/V weights as three separate linear projections.
  QKVProjectionOp(runtime::DeviceTensorView<const __nv_bfloat16> q_weight,
                  runtime::DeviceTensorView<const __nv_bfloat16> k_weight,
                  runtime::DeviceTensorView<const __nv_bfloat16> v_weight);

  // Projects X[rows, in_dim] into separate contiguous Q, K, and V outputs.
  void forward(const runtime::CudaContext& context,
               runtime::DeviceTensorView<const float> input,
               runtime::DeviceTensorView<float> q,
               runtime::DeviceTensorView<float> k,
               runtime::DeviceTensorView<float> v) const;

  size_t in_dim() const;
  size_t q_dim() const;
  size_t k_dim() const;
  size_t v_dim() const;

 private:
  runtime::DeviceTensorView<const __nv_bfloat16> q_weight_;
  runtime::DeviceTensorView<const __nv_bfloat16> k_weight_;
  runtime::DeviceTensorView<const __nv_bfloat16> v_weight_;
  LinearOp q_proj_;
  LinearOp k_proj_;
  LinearOp v_proj_;
};
