#pragma once

#include "config.h"
#include "operators/attention_op.h"
#include "operators/ffn_op.h"
#include "operators/linear_op.h"
#include "operators/qkv_projection_op.h"
#include "operators/residual_add_op.h"
#include "operators/rmsnorm_op.h"
#include "operators/rope_op.h"
#include "runtime/cuda_context.h"
#include "runtime/model_weights_gpu.h"
#include "runtime/rope_tables.h"
#include "runtime/workspace.h"

class DecoderBlock {
 public:
  // Binds one layer's weights and the shared RoPE tables.
  DecoderBlock(const runtime::LayerWeightsGPU& weights, const runtime::RoPETables& rope_tables,
               size_t num_q_heads,
               size_t num_kv_heads, size_t head_dim,
               float rms_norm_epsilon = RMS_NORM_EPSILON);

  // Runs one pre-norm decoder block and writes the final residual result to output.
  void forward(const runtime::CudaContext& context, runtime::Workspace& workspace,
               runtime::DeviceTensorView<const float> input,
               runtime::DeviceTensorView<float> output,
               size_t position_offset = 0) const;

  size_t hidden_dim() const;
  size_t q_dim() const;
  size_t kv_dim() const;

 private:
  RmsNormOp input_norm_;
  QKVProjectionOp qkv_proj_;
  RoPEOp rope_;
  AttentionOp attention_;
  LinearOp o_proj_;
  ResidualAddOp residual_add_;
  RmsNormOp post_attention_norm_;
  FFNOp ffn_;
};
