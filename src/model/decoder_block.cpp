#include "model/decoder_block.h"

namespace {

void validate_decoder_tensor(const runtime::DeviceTensorView<const float>& tensor,
                             const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("DecoderBlock: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("DecoderBlock: " + name + " must be rank-2");
}

void validate_decoder_tensor(const runtime::DeviceTensorView<float>& tensor,
                             const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("DecoderBlock: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("DecoderBlock: " + name + " must be rank-2");
}

}  // namespace

DecoderBlock::DecoderBlock(const runtime::LayerWeightsGPU& weights,
                           const runtime::RoPETables& rope_tables, size_t num_q_heads,
                           size_t num_kv_heads, size_t head_dim, float rms_norm_epsilon)
    : input_norm_(weights.input_layernorm.view(), rms_norm_epsilon),
      qkv_proj_(weights.q_proj.view(), weights.k_proj.view(), weights.v_proj.view()),
      rope_(rope_tables.cos(), rope_tables.sin(), num_q_heads, num_kv_heads, head_dim),
      attention_(num_q_heads, num_kv_heads, head_dim),
      o_proj_(weights.o_proj.view()),
      post_attention_norm_(weights.post_attention_layernorm.view(), rms_norm_epsilon),
      ffn_(weights.gate_proj.view(), weights.up_proj.view(), weights.down_proj.view()) {
  if (weights.input_layernorm.meta.shape.size() != 1 ||
      weights.post_attention_layernorm.meta.shape.size() != 1) {
    throw runtime_error("DecoderBlock: layernorm weights must be rank-1");
  }
  if (weights.input_layernorm.meta.shape[0] != weights.o_proj.meta.shape[0] ||
      weights.post_attention_layernorm.meta.shape[0] != weights.o_proj.meta.shape[0]) {
    throw runtime_error("DecoderBlock: layernorm widths must match hidden dimension");
  }
}

void DecoderBlock::forward(const runtime::CudaContext& context, runtime::Workspace& workspace,
                           runtime::DeviceTensorView<const float> input,
                           runtime::DeviceTensorView<float> output,
                           size_t position_offset) const {
  validate_decoder_tensor(input, "input");
  validate_decoder_tensor(output, "output");

  if (input.shape[1] != hidden_dim())
    throw runtime_error("DecoderBlock: input hidden dimension mismatch");
  if (output.shape != input.shape)
    throw runtime_error("DecoderBlock: output shape must match input shape");

  const size_t num_tokens = input.shape[0];

  runtime::DeviceTensorView<float> attn_norm =
      workspace.get_view("decoder_attn_norm", {num_tokens, hidden_dim()});
  runtime::DeviceTensorView<float> q =
      workspace.get_view("decoder_q", {num_tokens, q_dim()});
  runtime::DeviceTensorView<float> k =
      workspace.get_view("decoder_k", {num_tokens, kv_dim()});
  runtime::DeviceTensorView<float> v =
      workspace.get_view("decoder_v", {num_tokens, kv_dim()});
  runtime::DeviceTensorView<float> attn_out =
      workspace.get_view("decoder_attn_out", {num_tokens, q_dim()});
  runtime::DeviceTensorView<float> attn_proj =
      workspace.get_view("decoder_attn_proj", {num_tokens, hidden_dim()});
  runtime::DeviceTensorView<float> ffn_norm =
      workspace.get_view("decoder_ffn_norm", {num_tokens, hidden_dim()});

  input_norm_.forward(context, input, attn_norm);
  qkv_proj_.forward(context, runtime::DeviceTensorView<const float>(attn_norm.data, attn_norm.shape),
                    q, k, v);
  rope_.forward(context, q, k, position_offset);
  attention_.forward(context, workspace, runtime::DeviceTensorView<const float>(q.data, q.shape),
                     runtime::DeviceTensorView<const float>(k.data, k.shape),
                     runtime::DeviceTensorView<const float>(v.data, v.shape), attn_out,
                     position_offset, position_offset);
  o_proj_.forward(context, runtime::DeviceTensorView<const float>(attn_out.data, attn_out.shape),
                  attn_proj);
  residual_add_.forward_inplace(context, attn_proj, input);
  post_attention_norm_.forward(
      context, runtime::DeviceTensorView<const float>(attn_proj.data, attn_proj.shape),
      ffn_norm);
  ffn_.forward(context, workspace,
               runtime::DeviceTensorView<const float>(ffn_norm.data, ffn_norm.shape), output);
  residual_add_.forward_inplace(
      context, output, runtime::DeviceTensorView<const float>(attn_proj.data, attn_proj.shape));
}

size_t DecoderBlock::hidden_dim() const { return o_proj_.out_dim(); }

size_t DecoderBlock::q_dim() const { return qkv_proj_.q_dim(); }

size_t DecoderBlock::kv_dim() const { return qkv_proj_.k_dim(); }
