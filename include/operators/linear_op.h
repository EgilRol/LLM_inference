#pragma once

#include "operators/matmul_op.h"
#include "runtime/cuda_context.h"
#include "runtime/device_buffer.h"
#include "runtime/device_tensor.h"

class LinearOp {
 public:
  // Binds a checkpoint-layout weight [out_dim, in_dim] and stores a transposed copy once.
  LinearOp(const runtime::CudaContext& context,
           runtime::DeviceTensorView<const float> weight);

  // Applies X[rows, in_dim] * W^T[out_dim] and writes Y[rows, out_dim].
  void forward(const runtime::CudaContext& context,
               runtime::DeviceTensorView<const float> input,
               runtime::DeviceTensorView<float> output) const;

  runtime::DeviceTensorView<const float> weight() const;
  runtime::DeviceTensorView<const float> transposed_weight() const;
  size_t in_dim() const;
  size_t out_dim() const;

 private:
  runtime::DeviceTensorView<const float> weight_;
  runtime::DeviceBuffer<float> transposed_weight_;
  MatmulOp matmul_op_;
};
