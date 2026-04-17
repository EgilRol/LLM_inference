#include "test_api.h"
#include "config.h"
#include "io/staged_reader.h"
#include "io/weight_loader.h"
#include "kernel/kernels.cuh"
#include "operators/embedding_op.h"
#include "operators/matmul_op.h"
#include "runtime/device_buffer.h"
#include "runtime/cuda_compat.h"
#include "tokenizer.h"
#include <vector>

vector<int> TestAPI::tokenize(string input) {
  // Your code for testing the tokenizer goes here.
  // You can use the tokenizer you implemented in src/tokenizer_bpe.cpp and
  // include the header file here. Read the project description for more
  // information.
  BPETokenizer tok(TOKENIZER_PATH);
  vector<int> ids = tok.encode(input);
  ids.insert(ids.begin(), tok.bos_id_);

  return ids;
}

string TestAPI::detokenize(vector<int> token_ids) {
  BPETokenizer tok(TOKENIZER_PATH);
  string out = tok.decode(token_ids);
  return out;
}

vector<float> TestAPI::get_embeddings(vector<int> token_ids) {
  runtime::CudaContext context;
  io::WeightLoader embedding_loader(EMBED_WEIGHTS_PATH);
  io::StagedReader staged_reader(1 << 20);

  const io::TensorMeta& embed_meta = embedding_loader.meta("model.embed_tokens.weight");
  runtime::DeviceBuffer<__nv_bfloat16> embedding_table(embed_meta.num_elements);
  staged_reader.upload_tensor(embedding_loader, "model.embed_tokens.weight",
                              embedding_table.data(), context.stream());
  context.synchronize();

  EmbeddingOp embedding_op(runtime::make_device_tensor_view<const __nv_bfloat16>(
      embedding_table.data(), {embed_meta.shape[0], embed_meta.shape[1]}));

  runtime::DeviceBuffer<float> output(token_ids.size() * EMBEDDING_DIM);
  embedding_op.forward(
      context, token_ids,
      output.view({static_cast<size_t>(token_ids.size()), static_cast<size_t>(EMBEDDING_DIM)}));

  vector<float> host_output(token_ids.size() * EMBEDDING_DIM);
  output.copy_to_host(host_output.data(), host_output.size(), context.stream());
  context.synchronize();
  return host_output;
}

vector<float> TestAPI::matmul(const vector<float> &A, const vector<float> &B,
                              int M, int K, int N) {
  runtime::CudaContext context;
  MatmulOp matmul_op;

  runtime::DeviceBuffer<float> A_d(static_cast<size_t>(M) * K);
  runtime::DeviceBuffer<float> B_d(static_cast<size_t>(K) * N);
  runtime::DeviceBuffer<float> C_d(static_cast<size_t>(M) * N);

  A_d.copy_from_host(A.data(), A_d.size(), context.stream());
  B_d.copy_from_host(B.data(), B_d.size(), context.stream());

  matmul_op.forward(
      context,
      runtime::make_device_tensor_view<const float>(A_d.data(),
                                                    {static_cast<size_t>(M), static_cast<size_t>(K)}),
      runtime::make_device_tensor_view<const float>(B_d.data(),
                                                    {static_cast<size_t>(K), static_cast<size_t>(N)}),
      runtime::make_device_tensor_view<float>(C_d.data(),
                                              {static_cast<size_t>(M), static_cast<size_t>(N)}));

  vector<float> host_output(static_cast<size_t>(M) * N);
  C_d.copy_to_host(host_output.data(), host_output.size(), context.stream());
  context.synchronize();
  return host_output;
}
