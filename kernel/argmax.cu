#include "kernels.cuh"

#include <cfloat>

#include "runtime/cuda_utils.h"

__global__ void argmax_kernel(const float* input, int* output, int rows, int cols) {
  __shared__ float partial_max[ARGMAX_BLOCK_SIZE];
  __shared__ int partial_idx[ARGMAX_BLOCK_SIZE];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows)
    return;

  float local_max = -FLT_MAX;
  int local_idx = 0;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float value = input[row * cols + col];
    if (value > local_max || (value == local_max && col < local_idx)) {
      local_max = value;
      local_idx = col;
    }
  }

  partial_max[tid] = local_max;
  partial_idx[tid] = local_idx;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other_max = partial_max[tid + stride];
      const int other_idx = partial_idx[tid + stride];
      if (other_max > partial_max[tid] ||
          (other_max == partial_max[tid] && other_idx < partial_idx[tid])) {
        partial_max[tid] = other_max;
        partial_idx[tid] = other_idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0)
    output[row] = partial_idx[0];
}

void launch_argmax(const runtime::CudaContext& context,
                   runtime::DeviceTensorView<const float> input,
                   runtime::DeviceTensorView<int> output) {
  if (input.shape.size() != 2)
    throw runtime_error("launch_argmax: input must be rank-2");
  if (output.shape.size() != 1)
    throw runtime_error("launch_argmax: output must be rank-1");
  if (output.shape[0] != input.shape[0])
    throw runtime_error("launch_argmax: output length must match input row count");

  const int rows = static_cast<int>(input.shape[0]);
  const int cols = static_cast<int>(input.shape[1]);
  argmax_kernel<<<rows, ARGMAX_BLOCK_SIZE, 0, context.stream()>>>(input.data, output.data, rows,
                                                                  cols);
  runtime::check_last_launch("argmax_kernel launch");
}
