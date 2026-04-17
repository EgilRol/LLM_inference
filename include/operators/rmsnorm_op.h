#pragma once

#include "runtime/cuda_context.h"
#include "runtime/device_tensor.h"

class RmsNormOp {
 public:
  // Binds one learned gamma vector to a reusable RMSNorm operator instance.
  RmsNormOp(runtime::DeviceTensorView<const __nv_bfloat16> gamma, float epsilon);

  // Normalizes a [rows, cols] tensor row-wise and writes into the caller-owned output.
  void forward(const runtime::CudaContext& context,
               runtime::DeviceTensorView<const float> input,
               runtime::DeviceTensorView<float> output) const;

  runtime::DeviceTensorView<const __nv_bfloat16> gamma() const;
  float epsilon() const;

 private:
  runtime::DeviceTensorView<const __nv_bfloat16> gamma_;
  float epsilon_;
};
