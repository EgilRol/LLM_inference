#pragma once

#include "runtime/cuda_context.h"
#include "runtime/device_tensor.h"

class SwiGLUOp {
 public:
  // Applies SiLU(gate) * up elementwise over same-shaped [rows, cols] tensors.
  void forward(const runtime::CudaContext& context,
               runtime::DeviceTensorView<const float> gate,
               runtime::DeviceTensorView<const float> up,
               runtime::DeviceTensorView<float> output) const;
};
