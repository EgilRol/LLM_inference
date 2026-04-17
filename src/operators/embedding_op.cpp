#include "operators/embedding_op.h"

#include "kernel/kernels.cuh"

EmbeddingOp::EmbeddingOp(runtime::DeviceTensorView<const __nv_bfloat16> embedding_table)
    : embedding_table_(std::move(embedding_table)) {
  if (embedding_table_.data == nullptr)
    throw runtime_error("EmbeddingOp: embedding table data pointer is null");
  if (embedding_table_.shape.size() != 2)
    throw runtime_error("EmbeddingOp: embedding table must be rank-2");
}

void EmbeddingOp::forward(const runtime::CudaContext& context, const vector<int>& token_ids,
                          runtime::DeviceTensorView<float> output) const {
  if (output.data == nullptr)
    throw runtime_error("EmbeddingOp: output data pointer is null");
  if (output.shape.size() != 2)
    throw runtime_error("EmbeddingOp: output must be rank-2");
  if (output.shape[0] != token_ids.size() || output.shape[1] != hidden_dim())
    throw runtime_error("EmbeddingOp: output shape does not match token ids and hidden dim");

  for (int token_id : token_ids) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size())
      throw runtime_error("EmbeddingOp: token id out of embedding table bounds");
  }

  token_ids_buffer_.resize(token_ids.size());
  if (!token_ids.empty())
    token_ids_buffer_.copy_from_host(token_ids.data(), token_ids.size(), context.stream());

  launch_embedding_gather(
      context,
      runtime::make_device_tensor_view<const int>(token_ids_buffer_.data(), {token_ids.size()}),
      embedding_table_, output);
}

runtime::DeviceTensorView<const __nv_bfloat16> EmbeddingOp::embedding_table() const {
  return embedding_table_;
}

size_t EmbeddingOp::hidden_dim() const { return embedding_table_.shape[1]; }

size_t EmbeddingOp::vocab_size() const { return embedding_table_.shape[0]; }
