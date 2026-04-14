#include "kernels.cuh"

#include "runtime/cuda_utils.h"

__global__ void embedding_gather_kernel(const int* token_ids, const float* embedding_table,
                                        float* output, int num_tokens, int hidden_dim) {
  const int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_tokens * hidden_dim;
  if (flat_idx >= total)
    return;

  const int token_pos = flat_idx / hidden_dim;
  const int hidden_idx = flat_idx % hidden_dim;
  const int token_id = token_ids[token_pos];
  output[flat_idx] = embedding_table[token_id * hidden_dim + hidden_idx];
}

void launch_embedding_gather(const runtime::CudaContext& context,
                             runtime::DeviceTensorView<const int> token_ids,
                             runtime::DeviceTensorView<const float> embedding_table,
                             runtime::DeviceTensorView<float> output) {
  if (token_ids.shape.size() != 1)
    throw runtime_error("launch_embedding_gather: token ids must be rank-1");
  if (embedding_table.shape.size() != 2 || output.shape.size() != 2)
    throw runtime_error("launch_embedding_gather: expected rank-2 embedding table/output");
  if (output.shape[0] != token_ids.shape[0] || output.shape[1] != embedding_table.shape[1]) {
    throw runtime_error("launch_embedding_gather: output shape does not match embedding shapes");
  }

  const int num_tokens = static_cast<int>(token_ids.shape[0]);
  const int hidden_dim = static_cast<int>(embedding_table.shape[1]);
  const int total = num_tokens * hidden_dim;
  const int grid_dim = (total + EMBEDDING_BLOCK_SIZE - 1) / EMBEDDING_BLOCK_SIZE;
  embedding_gather_kernel<<<grid_dim, EMBEDDING_BLOCK_SIZE, 0, context.stream()>>>(
      token_ids.data, embedding_table.data, output.data, num_tokens, hidden_dim);
  runtime::check_last_launch("embedding_gather_kernel launch");
}
