#pragma once

#include "config.h"
#include "model/decoder_block.h"
#include "operators/argmax_op.h"
#include "operators/embedding_op.h"
#include "operators/linear_op.h"
#include "operators/rmsnorm_op.h"
#include "runtime/model_weights_gpu.h"
#include "runtime/rope_tables.h"
#include "runtime/workspace.h"
#include "tokenizer.h"

class LlamaModel {
 public:
  LlamaModel(const std::string& tokenizer_path, const std::string& weights_dir);

  std::string generate(const std::string& prompt, size_t max_new_tokens) const;

  vector<int> tokenize(const std::string& text) const;
  std::string detokenize(const vector<int>& token_ids) const;

 private:
  static constexpr size_t kNumQHeads = 32;
  static constexpr size_t kNumKVHeads = 8;
  static constexpr size_t kHeadDim = 128;
  static constexpr size_t kMaxPositions = 1024;

  void validate_model_geometry() const;
  int choose_next_token(runtime::DeviceTensorView<const float> logits) const;

  BPETokenizer tokenizer_;
  runtime::CudaContext context_;
  io::StagedReader staged_reader_;
  runtime::ModelWeightsGPU weights_;
  runtime::RoPETables rope_tables_;
  EmbeddingOp embedding_;
  vector<DecoderBlock> blocks_;
  RmsNormOp final_norm_;
  LinearOp lm_head_;
  ArgmaxOp argmax_;
  mutable runtime::Workspace workspace_;
  mutable runtime::DeviceBuffer<int> next_token_buffer_;
  size_t hidden_dim_;
  size_t vocab_size_;
};
