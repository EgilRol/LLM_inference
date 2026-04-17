#pragma once

#include "runtime/cuda_context.h"
#include "runtime/device_tensor.h"

class ResidualAddOp {
 public:
  // Adds two same-shaped [rows, cols] tensors elementwise.
  void forward(const runtime::CudaContext& context,
               runtime::DeviceTensorView<const float> input,
               runtime::DeviceTensorView<const float> residual,
               runtime::DeviceTensorView<float> output) const;

  // Decoder blocks usually overwrite the main activation with the residual sum.
  void forward_inplace(const runtime::CudaContext& context,
                       runtime::DeviceTensorView<float> input_output,
                       runtime::DeviceTensorView<const float> residual) const;
};
