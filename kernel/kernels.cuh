#pragma once

#include "prelude.h"
#include "runtime/cuda_context.h"
#include "runtime/device_tensor.h"
#include "runtime/cuda_compat.h"

constexpr int TILE_SIZE = 32;
constexpr int RMSNORM_BLOCK_SIZE = 256;
constexpr int EMBEDDING_BLOCK_SIZE = 256;

#if LLM_RUNTIME_HAS_CUDA
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N);
__global__ void rmsnorm_kernel(const float* input, const float* gamma, float* output,
                               int rows, int cols, float epsilon);
__global__ void embedding_gather_kernel(const int* token_ids, const float* embedding_table,
                                        float* output, int num_tokens, int hidden_dim);
__global__ void transpose_kernel(const float* input, float* output, int rows, int cols);
#endif

void launch_matmul(const runtime::CudaContext& context,
                   runtime::DeviceTensorView<const float> A,
                   runtime::DeviceTensorView<const float> B,
                   runtime::DeviceTensorView<float> C);
void launch_transpose(const runtime::CudaContext& context,
                      runtime::DeviceTensorView<const float> input,
                      runtime::DeviceTensorView<float> output);
void launch_rmsnorm(const runtime::CudaContext& context,
                    runtime::DeviceTensorView<const float> input,
                    runtime::DeviceTensorView<const float> gamma,
                    runtime::DeviceTensorView<float> output, float epsilon);
void launch_embedding_gather(const runtime::CudaContext& context,
                             runtime::DeviceTensorView<const int> token_ids,
                             runtime::DeviceTensorView<const float> embedding_table,
                             runtime::DeviceTensorView<float> output);
