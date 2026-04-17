#include "kernels.cuh"

#include "runtime/cuda_utils.h"

__global__ void residual_add_kernel(const float* input, const float* residual, float* output,
                                    int total) {
  const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (flat_idx >= total)
    return;

  output[flat_idx] = input[flat_idx] + residual[flat_idx];
}

void launch_residual_add(const runtime::CudaContext& context,
                         runtime::DeviceTensorView<const float> input,
                         runtime::DeviceTensorView<const float> residual,
                         runtime::DeviceTensorView<float> output) {
  if (input.shape.size() != 2 || residual.shape.size() != 2 || output.shape.size() != 2) {
    throw runtime_error("launch_residual_add: expected rank-2 tensors");
  }
  if (input.shape != residual.shape || input.shape != output.shape) {
    throw runtime_error("launch_residual_add: input, residual, and output shapes must match");
  }

  const int total = static_cast<int>(input.num_elements());
  const int grid_dim = (total + RESIDUAL_ADD_BLOCK_SIZE - 1) / RESIDUAL_ADD_BLOCK_SIZE;
  residual_add_kernel<<<grid_dim, RESIDUAL_ADD_BLOCK_SIZE, 0, context.stream()>>>(
      input.data, residual.data, output.data, total);
  runtime::check_last_launch("residual_add_kernel launch");
}
