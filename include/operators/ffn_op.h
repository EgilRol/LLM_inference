#pragma once

#include "operators/linear_op.h"
#include "operators/swiglu_op.h"
#include "runtime/cuda_context.h"
#include "runtime/device_tensor.h"
#include "runtime/workspace.h"

class FFNOp {
 public:
  // Binds the three FFN projections and reuses workspace for intermediate activations.
  FFNOp(runtime::DeviceTensorView<const __nv_bfloat16> gate_weight,
        runtime::DeviceTensorView<const __nv_bfloat16> up_weight,
        runtime::DeviceTensorView<const __nv_bfloat16> down_weight);

  void forward(const runtime::CudaContext& context, runtime::Workspace& workspace,
               runtime::DeviceTensorView<const float> input,
               runtime::DeviceTensorView<float> output) const;

  size_t in_dim() const;
  size_t hidden_dim() const;
  size_t out_dim() const;

 private:
  runtime::DeviceTensorView<const __nv_bfloat16> gate_weight_;
  runtime::DeviceTensorView<const __nv_bfloat16> up_weight_;
  runtime::DeviceTensorView<const __nv_bfloat16> down_weight_;
  LinearOp gate_proj_;
  LinearOp up_proj_;
  LinearOp down_proj_;
  SwiGLUOp swiglu_;
};
