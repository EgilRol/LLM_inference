#include "kernels.cuh"

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int K,
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

vector<float> matmul(const vector<float> &A_h, const vector<float> &B_h, int M,
                     int K, int N) {
  size_t size_A = static_cast<size_t>(M) * K * sizeof(float);
  size_t size_B = static_cast<size_t>(K) * N * sizeof(float);
  size_t size_C = static_cast<size_t>(M) * N * sizeof(float);

  float *A_d = nullptr;
  float *B_d = nullptr;
  float *C_d = nullptr;

  cudaMalloc((void **)&A_d, size_A);
  cudaMalloc((void **)&B_d, size_B);
  cudaMalloc((void **)&C_d, size_C);

  cudaMemcpy(A_d, A_h.data(), size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h.data(), size_B, cudaMemcpyHostToDevice);

  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
               (M + TILE_SIZE - 1) / TILE_SIZE);

  matmul_kernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, K, N);

  vector<float> C_h(static_cast<size_t>(M) * N);
  cudaMemcpy(C_h.data(), C_d, size_C, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return C_h;
}
