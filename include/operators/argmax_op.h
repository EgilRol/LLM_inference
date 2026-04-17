#pragma once

#include "runtime/cuda_context.h"
#include "runtime/device_tensor.h"

class ArgmaxOp {
 public:
  // Computes argmax over the last dimension of a rank-2 input and writes one int per row.
  void forward(const runtime::CudaContext& context,
               runtime::DeviceTensorView<const float> input,
               runtime::DeviceTensorView<int> output) const;
};
