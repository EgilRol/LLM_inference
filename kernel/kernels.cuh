#pragma once

#include "prelude.h"
#include "runtime/cuda_context.h"
#include "runtime/device_tensor.h"
#include "runtime/cuda_compat.h"

constexpr int TILE_SIZE = 32;
constexpr int RMSNORM_BLOCK_SIZE = 256;
constexpr int EMBEDDING_BLOCK_SIZE = 256;
constexpr int ROPE_BLOCK_SIZE = 256;
constexpr int ATTENTION_SCORE_BLOCK_SIZE = 256;
constexpr int SOFTMAX_BLOCK_SIZE = 256;
constexpr int ATTENTION_WEIGHTED_SUM_BLOCK_SIZE = 256;
constexpr int RESIDUAL_ADD_BLOCK_SIZE = 256;
constexpr int SWIGLU_BLOCK_SIZE = 256;
constexpr int ARGMAX_BLOCK_SIZE = 256;

#if LLM_RUNTIME_HAS_CUDA
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N);
__global__ void linear_matmul_kernel(const float* input, const __nv_bfloat16* weight,
                                     float* output, int rows, int in_dim, int out_dim);
__global__ void rmsnorm_kernel(const float* input, const __nv_bfloat16* gamma, float* output,
                               int rows, int cols, float epsilon);
__global__ void embedding_gather_kernel(const int* token_ids,
                                        const __nv_bfloat16* embedding_table,
                                        float* output, int num_tokens, int hidden_dim);
__global__ void rope_kernel(float* tensor, const float* cos_table, const float* sin_table,
                            int num_tokens, int num_heads, int head_dim, int position_offset);
__global__ void residual_add_kernel(const float* input, const float* residual, float* output,
                                    int total);
__global__ void swiglu_kernel(const float* gate, const float* up, float* output, int total);
__global__ void argmax_kernel(const float* input, int* output, int rows, int cols);
__global__ void attention_score_kernel(const float* q, const float* k, float* scores,
                                       int q_tokens, int k_tokens, int num_q_heads,
                                       int num_kv_heads, int head_dim, int q_pos_offset,
                                       int k_pos_offset, float scale);
__global__ void softmax_kernel(float* scores, int rows, int cols);
__global__ void attention_weighted_sum_kernel(const float* scores, const float* v, float* output,
                                              int q_tokens, int k_tokens, int num_q_heads,
                                              int num_kv_heads, int head_dim);
#endif

void launch_matmul(const runtime::CudaContext& context,
                   runtime::DeviceTensorView<const float> A,
                   runtime::DeviceTensorView<const float> B,
                   runtime::DeviceTensorView<float> C);
void launch_linear_matmul(const runtime::CudaContext& context,
                          runtime::DeviceTensorView<const float> input,
                          runtime::DeviceTensorView<const __nv_bfloat16> weight,
                          runtime::DeviceTensorView<float> output);
void launch_rmsnorm(const runtime::CudaContext& context,
                    runtime::DeviceTensorView<const float> input,
                    runtime::DeviceTensorView<const __nv_bfloat16> gamma,
                    runtime::DeviceTensorView<float> output, float epsilon);
void launch_embedding_gather(const runtime::CudaContext& context,
                             runtime::DeviceTensorView<const int> token_ids,
                             runtime::DeviceTensorView<const __nv_bfloat16> embedding_table,
                             runtime::DeviceTensorView<float> output);
void launch_residual_add(const runtime::CudaContext& context,
                         runtime::DeviceTensorView<const float> input,
                         runtime::DeviceTensorView<const float> residual,
                         runtime::DeviceTensorView<float> output);
void launch_swiglu(const runtime::CudaContext& context,
                   runtime::DeviceTensorView<const float> gate,
                   runtime::DeviceTensorView<const float> up,
                   runtime::DeviceTensorView<float> output);
void launch_argmax(const runtime::CudaContext& context,
                   runtime::DeviceTensorView<const float> input,
                   runtime::DeviceTensorView<int> output);
void launch_rope(const runtime::CudaContext& context, runtime::DeviceTensorView<float> tensor,
                 runtime::DeviceTensorView<const float> cos_table,
                 runtime::DeviceTensorView<const float> sin_table, size_t num_heads,
                 size_t head_dim, size_t position_offset);
void launch_attention_scores(const runtime::CudaContext& context,
                             runtime::DeviceTensorView<const float> q,
                             runtime::DeviceTensorView<const float> k,
                             runtime::DeviceTensorView<float> scores, size_t num_q_heads,
                             size_t num_kv_heads, size_t head_dim, size_t q_position_offset,
                             size_t k_position_offset, float scale);
void launch_softmax(const runtime::CudaContext& context, runtime::DeviceTensorView<float> scores);
void launch_attention_weighted_sum(const runtime::CudaContext& context,
                                   runtime::DeviceTensorView<const float> scores,
                                   runtime::DeviceTensorView<const float> v,
                                   runtime::DeviceTensorView<float> output, size_t num_q_heads,
                                   size_t num_kv_heads, size_t head_dim);
