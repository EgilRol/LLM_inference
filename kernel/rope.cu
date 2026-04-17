#include "kernels.cuh"

#include "runtime/cuda_utils.h"

__global__ void rope_kernel(float* tensor, const float* cos_table, const float* sin_table,
                            int num_tokens, int num_heads, int head_dim, int position_offset) {
  const int pair_dim = head_dim / 2;
  const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_tokens * num_heads * pair_dim;
  if (flat_idx >= total)
    return;

  const int pair_idx = flat_idx % pair_dim;
  const int head_idx = (flat_idx / pair_dim) % num_heads;
  const int token_idx = flat_idx / (num_heads * pair_dim);

  const int row_width = num_heads * head_dim;
  const int base = token_idx * row_width + head_idx * head_dim;
  const int upper_idx = base + pair_idx;
  const int lower_idx = base + pair_dim + pair_idx;
  const int table_idx = (token_idx + position_offset) * pair_dim + pair_idx;

  const float cos_v = cos_table[table_idx];
  const float sin_v = sin_table[table_idx];
  const float upper = tensor[upper_idx];
  const float lower = tensor[lower_idx];

  tensor[upper_idx] = upper * cos_v - lower * sin_v;
  tensor[lower_idx] = lower * cos_v + upper * sin_v;
}

void launch_rope(const runtime::CudaContext& context, runtime::DeviceTensorView<float> tensor,
                 runtime::DeviceTensorView<const float> cos_table,
                 runtime::DeviceTensorView<const float> sin_table, size_t num_heads,
                 size_t head_dim, size_t position_offset) {
  if (tensor.shape.size() != 2)
    throw runtime_error("launch_rope: tensor must be rank-2");
  if (cos_table.shape.size() != 2 || sin_table.shape.size() != 2)
    throw runtime_error("launch_rope: cos/sin tables must be rank-2");
  if (cos_table.shape != sin_table.shape)
    throw runtime_error("launch_rope: cos and sin table shapes must match");
  if (head_dim == 0 || head_dim % 2 != 0)
    throw runtime_error("launch_rope: head_dim must be positive and even");
  if (tensor.shape[1] != num_heads * head_dim)
    throw runtime_error("launch_rope: tensor width does not match num_heads * head_dim");
  if (cos_table.shape[1] != head_dim / 2)
    throw runtime_error("launch_rope: table width must equal head_dim / 2");
  if (position_offset + tensor.shape[0] > cos_table.shape[0])
    throw runtime_error("launch_rope: requested positions exceed precomputed table");

  const int num_tokens = static_cast<int>(tensor.shape[0]);
  const int total = num_tokens * static_cast<int>(num_heads) * static_cast<int>(head_dim / 2);
  const int grid_dim = (total + ROPE_BLOCK_SIZE - 1) / ROPE_BLOCK_SIZE;
  rope_kernel<<<grid_dim, ROPE_BLOCK_SIZE, 0, context.stream()>>>(
      tensor.data, cos_table.data, sin_table.data, num_tokens, static_cast<int>(num_heads),
      static_cast<int>(head_dim), static_cast<int>(position_offset));
  runtime::check_last_launch("rope_kernel launch");
}
