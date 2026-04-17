#include "kernels.cuh"

#include <cfloat>

#include "runtime/cuda_utils.h"

namespace {

__device__ int kv_head_for_q_head(int q_head, int num_q_heads, int num_kv_heads) {
  const int group_size = num_q_heads / num_kv_heads;
  return q_head / group_size;
}

}  // namespace

__global__ void attention_score_kernel(const float* q, const float* k, float* scores,
                                       int q_tokens, int k_tokens, int num_q_heads,
                                       int num_kv_heads, int head_dim, int q_pos_offset,
                                       int k_pos_offset, float scale) {
  const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_q_heads * q_tokens * k_tokens;
  if (flat_idx >= total)
    return;

  const int k_token = flat_idx % k_tokens;
  const int q_token = (flat_idx / k_tokens) % q_tokens;
  const int q_head = flat_idx / (q_tokens * k_tokens);
  const int kv_head = kv_head_for_q_head(q_head, num_q_heads, num_kv_heads);

  const int q_abs_pos = q_token + q_pos_offset;
  const int k_abs_pos = k_token + k_pos_offset;
  const int score_row = q_head * q_tokens + q_token;
  const int score_idx = score_row * k_tokens + k_token;

  if (k_abs_pos > q_abs_pos) {
    scores[score_idx] = -FLT_MAX;
    return;
  }

  const int q_row_width = num_q_heads * head_dim;
  const int k_row_width = num_kv_heads * head_dim;
  const int q_base = q_token * q_row_width + q_head * head_dim;
  const int k_base = k_token * k_row_width + kv_head * head_dim;

  float dot = 0.0f;
  for (int dim = 0; dim < head_dim; ++dim)
    dot += q[q_base + dim] * k[k_base + dim];

  scores[score_idx] = dot * scale;
}

__global__ void softmax_kernel(float* scores, int rows, int cols) {
  __shared__ float partial_max[SOFTMAX_BLOCK_SIZE];
  __shared__ float partial_sum[SOFTMAX_BLOCK_SIZE];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows)
    return;

  float local_max = -FLT_MAX;
  for (int col = tid; col < cols; col += blockDim.x)
    local_max = fmaxf(local_max, scores[row * cols + col]);

  partial_max[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      partial_max[tid] = fmaxf(partial_max[tid], partial_max[tid + stride]);
    __syncthreads();
  }

  const float row_max = partial_max[0];
  float local_sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float value = expf(scores[row * cols + col] - row_max);
    scores[row * cols + col] = value;
    local_sum += value;
  }

  partial_sum[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      partial_sum[tid] += partial_sum[tid + stride];
    __syncthreads();
  }

  const float inv_sum = 1.0f / partial_sum[0];
  for (int col = tid; col < cols; col += blockDim.x)
    scores[row * cols + col] *= inv_sum;
}

__global__ void attention_weighted_sum_kernel(const float* scores, const float* v, float* output,
                                              int q_tokens, int k_tokens, int num_q_heads,
                                              int num_kv_heads, int head_dim) {
  const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = q_tokens * num_q_heads * head_dim;
  if (flat_idx >= total)
    return;

  const int dim = flat_idx % head_dim;
  const int q_head = (flat_idx / head_dim) % num_q_heads;
  const int q_token = flat_idx / (num_q_heads * head_dim);
  const int kv_head = kv_head_for_q_head(q_head, num_q_heads, num_kv_heads);

  const int score_row = q_head * q_tokens + q_token;
  const int v_row_width = num_kv_heads * head_dim;

  float weighted_sum = 0.0f;
  for (int k_token = 0; k_token < k_tokens; ++k_token) {
    const float attn = scores[score_row * k_tokens + k_token];
    const int v_idx = k_token * v_row_width + kv_head * head_dim + dim;
    weighted_sum += attn * v[v_idx];
  }

  const int out_row_width = num_q_heads * head_dim;
  output[q_token * out_row_width + q_head * head_dim + dim] = weighted_sum;
}

void launch_attention_scores(const runtime::CudaContext& context,
                             runtime::DeviceTensorView<const float> q,
                             runtime::DeviceTensorView<const float> k,
                             runtime::DeviceTensorView<float> scores, size_t num_q_heads,
                             size_t num_kv_heads, size_t head_dim, size_t q_position_offset,
                             size_t k_position_offset, float scale) {
  if (q.shape.size() != 2 || k.shape.size() != 2 || scores.shape.size() != 2)
    throw runtime_error("launch_attention_scores: expected rank-2 tensors");
  if (head_dim == 0)
    throw runtime_error("launch_attention_scores: head_dim must be positive");
  if (num_q_heads == 0 || num_kv_heads == 0)
    throw runtime_error("launch_attention_scores: head counts must be positive");
  if (num_q_heads % num_kv_heads != 0)
    throw runtime_error("launch_attention_scores: num_q_heads must be divisible by num_kv_heads");
  if (q.shape[1] != num_q_heads * head_dim)
    throw runtime_error("launch_attention_scores: Q width does not match num_q_heads * head_dim");
  if (k.shape[1] != num_kv_heads * head_dim)
    throw runtime_error("launch_attention_scores: K width does not match num_kv_heads * head_dim");
  if (scores.shape[0] != num_q_heads * q.shape[0] || scores.shape[1] != k.shape[0]) {
    throw runtime_error("launch_attention_scores: score tensor shape mismatch");
  }

  const int q_tokens = static_cast<int>(q.shape[0]);
  const int k_tokens = static_cast<int>(k.shape[0]);
  const int total = q_tokens * k_tokens * static_cast<int>(num_q_heads);
  const int grid_dim = (total + ATTENTION_SCORE_BLOCK_SIZE - 1) / ATTENTION_SCORE_BLOCK_SIZE;
  attention_score_kernel<<<grid_dim, ATTENTION_SCORE_BLOCK_SIZE, 0, context.stream()>>>(
      q.data, k.data, scores.data, q_tokens, k_tokens, static_cast<int>(num_q_heads),
      static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
      static_cast<int>(q_position_offset), static_cast<int>(k_position_offset), scale);
  runtime::check_last_launch("attention_score_kernel launch");
}

void launch_softmax(const runtime::CudaContext& context, runtime::DeviceTensorView<float> scores) {
  if (scores.shape.size() != 2)
    throw runtime_error("launch_softmax: expected a rank-2 tensor");

  const int rows = static_cast<int>(scores.shape[0]);
  const int cols = static_cast<int>(scores.shape[1]);
  softmax_kernel<<<rows, SOFTMAX_BLOCK_SIZE, 0, context.stream()>>>(scores.data, rows, cols);
  runtime::check_last_launch("softmax_kernel launch");
}

void launch_attention_weighted_sum(const runtime::CudaContext& context,
                                   runtime::DeviceTensorView<const float> scores,
                                   runtime::DeviceTensorView<const float> v,
                                   runtime::DeviceTensorView<float> output, size_t num_q_heads,
                                   size_t num_kv_heads, size_t head_dim) {
  if (scores.shape.size() != 2 || v.shape.size() != 2 || output.shape.size() != 2)
    throw runtime_error("launch_attention_weighted_sum: expected rank-2 tensors");
  if (head_dim == 0)
    throw runtime_error("launch_attention_weighted_sum: head_dim must be positive");
  if (num_q_heads == 0 || num_kv_heads == 0)
    throw runtime_error("launch_attention_weighted_sum: head counts must be positive");
  if (num_q_heads % num_kv_heads != 0) {
    throw runtime_error(
        "launch_attention_weighted_sum: num_q_heads must be divisible by num_kv_heads");
  }

  const size_t q_tokens = output.shape[0];
  const size_t k_tokens = v.shape[0];
  if (scores.shape[0] != num_q_heads * q_tokens || scores.shape[1] != k_tokens)
    throw runtime_error("launch_attention_weighted_sum: score tensor shape mismatch");
  if (v.shape[1] != num_kv_heads * head_dim)
    throw runtime_error("launch_attention_weighted_sum: V width mismatch");
  if (output.shape[1] != num_q_heads * head_dim)
    throw runtime_error("launch_attention_weighted_sum: output width mismatch");

  const int total = static_cast<int>(q_tokens * num_q_heads * head_dim);
  const int grid_dim =
      (total + ATTENTION_WEIGHTED_SUM_BLOCK_SIZE - 1) / ATTENTION_WEIGHTED_SUM_BLOCK_SIZE;
  attention_weighted_sum_kernel<<<grid_dim, ATTENTION_WEIGHTED_SUM_BLOCK_SIZE, 0,
                                  context.stream()>>>(
      scores.data, v.data, output.data, static_cast<int>(q_tokens), static_cast<int>(k_tokens),
      static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
      static_cast<int>(head_dim));
  runtime::check_last_launch("attention_weighted_sum_kernel launch");
}
