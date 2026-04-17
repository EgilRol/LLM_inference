#include "kernels.cuh"

#include "runtime/cuda_utils.h"

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int K,
                              int N) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  float sum = 0.0f;
  int numPhases = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int phase = 0; phase < numPhases; ++phase) {
    int kOff = phase * TILE_SIZE;

    if (row < M && kOff + tx < K)
      As[ty][tx] = A[row * K + kOff + tx];
    else
      As[ty][tx] = 0.0f;

    if (kOff + ty < K && col < N)
      Bs[ty][tx] = B[(kOff + ty) * N + col];
    else
      Bs[ty][tx] = 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k)
      sum += As[ty][k] * Bs[k][tx];

    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = sum;
}

__global__ void linear_matmul_kernel(const float* input, const __nv_bfloat16* weight,
                                     float* output, int rows, int in_dim, int out_dim) {
  __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
  __shared__ float weight_tile[TILE_SIZE][TILE_SIZE];

  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  const int row = blockIdx.y * TILE_SIZE + ty;
  const int out_col = blockIdx.x * TILE_SIZE + tx;

  float sum = 0.0f;
  const int num_phases = (in_dim + TILE_SIZE - 1) / TILE_SIZE;
  for (int phase = 0; phase < num_phases; ++phase) {
    const int k_off = phase * TILE_SIZE;

    if (row < rows && k_off + tx < in_dim)
      input_tile[ty][tx] = input[row * in_dim + k_off + tx];
    else
      input_tile[ty][tx] = 0.0f;

    if (out_col < out_dim && k_off + ty < in_dim) {
      weight_tile[ty][tx] =
          __bfloat162float(weight[out_col * in_dim + k_off + ty]);
    } else {
      weight_tile[ty][tx] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k)
      sum += input_tile[ty][k] * weight_tile[k][tx];

    __syncthreads();
  }

  if (row < rows && out_col < out_dim)
    output[row * out_dim + out_col] = sum;
}

void launch_matmul(const runtime::CudaContext& context,
                   runtime::DeviceTensorView<const float> A,
                   runtime::DeviceTensorView<const float> B,
                   runtime::DeviceTensorView<float> C) {
  if (A.shape.size() != 2 || B.shape.size() != 2 || C.shape.size() != 2)
    throw runtime_error("launch_matmul: expected rank-2 tensors");

  const int M = static_cast<int>(A.shape[0]);
  const int K = static_cast<int>(A.shape[1]);
  const int B_rows = static_cast<int>(B.shape[0]);
  const int N = static_cast<int>(B.shape[1]);

  if (B_rows != K)
    throw runtime_error("launch_matmul: inner dimensions do not match");
  if (static_cast<int>(C.shape[0]) != M || static_cast<int>(C.shape[1]) != N)
    throw runtime_error("launch_matmul: output shape does not match");

  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  matmul_kernel<<<grid_dim, block_dim, 0, context.stream()>>>(A.data, B.data, C.data, M, K, N);
  runtime::check_last_launch("matmul_kernel launch");
}

void launch_linear_matmul(const runtime::CudaContext& context,
                          runtime::DeviceTensorView<const float> input,
                          runtime::DeviceTensorView<const __nv_bfloat16> weight,
                          runtime::DeviceTensorView<float> output) {
  if (input.shape.size() != 2 || weight.shape.size() != 2 || output.shape.size() != 2)
    throw runtime_error("launch_linear_matmul: expected rank-2 tensors");
  if (input.shape[1] != weight.shape[1])
    throw runtime_error("launch_linear_matmul: input hidden dim does not match weight");
  if (output.shape[0] != input.shape[0] || output.shape[1] != weight.shape[0])
    throw runtime_error("launch_linear_matmul: output shape does not match");

  const int rows = static_cast<int>(input.shape[0]);
  const int in_dim = static_cast<int>(input.shape[1]);
  const int out_dim = static_cast<int>(weight.shape[0]);
  dim3 block_dim(TILE_SIZE, TILE_SIZE);
  dim3 grid_dim((out_dim + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);
  linear_matmul_kernel<<<grid_dim, block_dim, 0, context.stream()>>>(
      input.data, weight.data, output.data, rows, in_dim, out_dim);
  runtime::check_last_launch("linear_matmul_kernel launch");
}
