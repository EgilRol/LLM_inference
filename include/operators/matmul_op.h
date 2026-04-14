#pragma once

#include "runtime/cuda_context.h"
#include "runtime/device_tensor.h"

class MatmulOp {
 public:
  // Multiplies A[M, K] by B[K, N] and writes the result into C[M, N].
  void forward(const runtime::CudaContext& context,
               runtime::DeviceTensorView<const float> A,
               runtime::DeviceTensorView<const float> B,
               runtime::DeviceTensorView<float> C) const;
};
