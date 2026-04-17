#include "kernels.cuh"

#include "runtime/cuda_utils.h"

__global__ void rmsnorm_kernel(const float* input, const __nv_bfloat16* gamma, float* output,
                               int rows, int cols, float epsilon) {
  __shared__ float partial_sums[RMSNORM_BLOCK_SIZE];
  __shared__ float inv_rms;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows)
    return;

  float sum_squares = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float value = input[row * cols + col];
    sum_squares += value * value;
  }
  partial_sums[tid] = sum_squares;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      partial_sums[tid] += partial_sums[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    inv_rms = rsqrtf(partial_sums[0] / static_cast<float>(cols) + epsilon);
  __syncthreads();

  for (int col = tid; col < cols; col += blockDim.x) {
    const float value = input[row * cols + col];
    output[row * cols + col] = value * inv_rms * __bfloat162float(gamma[col]);
  }
}

void launch_rmsnorm(const runtime::CudaContext& context,
                    runtime::DeviceTensorView<const float> input,
                    runtime::DeviceTensorView<const __nv_bfloat16> gamma,
                    runtime::DeviceTensorView<float> output, float epsilon) {
  if (input.shape.size() != 2 || output.shape.size() != 2)
    throw runtime_error("launch_rmsnorm: expected rank-2 input/output tensors");
  if (gamma.shape.size() != 1)
    throw runtime_error("launch_rmsnorm: expected rank-1 gamma tensor");
  if (input.shape != output.shape)
    throw runtime_error("launch_rmsnorm: input and output shapes must match");
  if (input.shape[1] != gamma.shape[0])
    throw runtime_error("launch_rmsnorm: hidden dimension does not match gamma");

  const int rows = static_cast<int>(input.shape[0]);
  const int cols = static_cast<int>(input.shape[1]);
  rmsnorm_kernel<<<rows, RMSNORM_BLOCK_SIZE, 0, context.stream()>>>(
      input.data, gamma.data, output.data, rows, cols, epsilon);
  runtime::check_last_launch("rmsnorm_kernel launch");
}
