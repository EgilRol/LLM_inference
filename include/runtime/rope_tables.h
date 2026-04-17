#pragma once

#include "runtime/cuda_context.h"
#include "runtime/device_buffer.h"

namespace runtime {

class RoPETables {
 public:
  // Builds cos/sin tables for positions [0, max_positions) and pair indices [0, head_dim/2).
  RoPETables(const CudaContext& context, size_t max_positions, size_t head_dim,
             float base);

  DeviceTensorView<const float> cos() const;
  DeviceTensorView<const float> sin() const;
  size_t max_positions() const;
  size_t head_dim() const;
  size_t pair_dim() const;
  float base() const;

 private:
  size_t max_positions_;
  size_t head_dim_;
  float base_;
  DeviceBuffer<float> cos_;
  DeviceBuffer<float> sin_;
};

}  // namespace runtime
