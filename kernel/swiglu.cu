#include "kernels.cuh"

#include "runtime/cuda_utils.h"

__global__ void swiglu_kernel(const float* gate, const float* up, float* output, int total) {
  const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx >= total)
    return;

  const float gate_value = gate[flat_idx];
  const float silu = gate_value / (1.0f + expf(-gate_value));
  output[flat_idx] = silu * up[flat_idx];
}

void launch_swiglu(const runtime::CudaContext& context,
                   runtime::DeviceTensorView<const float> gate,
                   runtime::DeviceTensorView<const float> up,
                   runtime::DeviceTensorView<float> output) {
  if (gate.shape.size() != 2 || up.shape.size() != 2 || output.shape.size() != 2)
    throw runtime_error("launch_swiglu: expected rank-2 tensors");
  if (gate.shape != up.shape || gate.shape != output.shape)
    throw runtime_error("launch_swiglu: gate, up, and output shapes must match");

  const int total = static_cast<int>(gate.num_elements());
  const int grid_dim = (total + SWIGLU_BLOCK_SIZE - 1) / SWIGLU_BLOCK_SIZE;
  swiglu_kernel<<<grid_dim, SWIGLU_BLOCK_SIZE, 0, context.stream()>>>(
      gate.data, up.data, output.data, total);
  runtime::check_last_launch("swiglu_kernel launch");
}
