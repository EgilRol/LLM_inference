#pragma once

#include "runtime/cuda_context.h"
#include "runtime/device_buffer.h"
#include "runtime/device_tensor.h"

class EmbeddingOp {
 public:
  // Binds the embedding table and keeps a small reusable device buffer for token ids.
  explicit EmbeddingOp(runtime::DeviceTensorView<const __nv_bfloat16> embedding_table);

  // Uploads token ids, gathers embedding rows on GPU, and writes into caller-owned output.
  void forward(const runtime::CudaContext& context, const vector<int>& token_ids,
               runtime::DeviceTensorView<float> output) const;

  runtime::DeviceTensorView<const __nv_bfloat16> embedding_table() const;
  size_t hidden_dim() const;
  size_t vocab_size() const;

 private:
  runtime::DeviceTensorView<const __nv_bfloat16> embedding_table_;
  mutable runtime::DeviceBuffer<int> token_ids_buffer_;
};
