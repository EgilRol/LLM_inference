#include "model/llama_model.h"

namespace {

runtime::DeviceTensorView<const float> const_view(runtime::DeviceTensorView<float> tensor) {
  return runtime::DeviceTensorView<const float>(tensor.data, tensor.shape);
}

}  // namespace

LlamaModel::LlamaModel(const std::string& tokenizer_path, const std::string& weights_dir)
    : tokenizer_(tokenizer_path),
      staged_reader_(1 << 20),
      weights_(weights_dir, staged_reader_, context_),
      rope_tables_(context_, kMaxPositions, kHeadDim, ROPE_BASE),
      embedding_(weights_.embed_tokens()),
      final_norm_(weights_.model_norm(), RMS_NORM_EPSILON),
      lm_head_(weights_.lm_head()),
      hidden_dim_(weights_.model_norm().shape[0]),
      vocab_size_(weights_.lm_head().shape[0]) {
  validate_model_geometry();

  blocks_.reserve(runtime::ModelWeightsGPU::kNumLayers);
  for (size_t layer_idx = 0; layer_idx < runtime::ModelWeightsGPU::kNumLayers; ++layer_idx) {
    blocks_.emplace_back(weights_.layer(layer_idx), rope_tables_, kNumQHeads, kNumKVHeads,
                         kHeadDim);
  }
}

std::string LlamaModel::generate(const std::string& prompt, size_t max_new_tokens) const {
  vector<int> token_ids = tokenize(prompt);
  if (token_ids.empty() || token_ids.front() != tokenizer_.bos_id())
    token_ids.insert(token_ids.begin(), tokenizer_.bos_id());

  if (token_ids.size() > kMaxPositions || token_ids.size() + max_new_tokens > kMaxPositions) {
    throw runtime_error("LlamaModel: requested sequence length exceeds max supported positions");
  }

  for (size_t step = 0; step < max_new_tokens; ++step) {
    const size_t seq_len = token_ids.size();

    auto& act_a_buffer = workspace_.get_buffer("llama_act_a", seq_len * hidden_dim_);
    auto& act_b_buffer = workspace_.get_buffer("llama_act_b", seq_len * hidden_dim_);
    runtime::DeviceTensorView<float> act_a = act_a_buffer.view({seq_len, hidden_dim_});
    runtime::DeviceTensorView<float> act_b = act_b_buffer.view({seq_len, hidden_dim_});

    embedding_.forward(context_, token_ids, act_a);

    runtime::DeviceTensorView<float> current = act_a;
    runtime::DeviceTensorView<float> next = act_b;
    for (const DecoderBlock& block : blocks_) {
      block.forward(context_, workspace_, const_view(current), next, 0);
      std::swap(current, next);
    }

    final_norm_.forward(context_, const_view(current), next);
    current = next;

    auto& logits_buffer = workspace_.get_buffer("llama_logits", vocab_size_);
    runtime::DeviceTensorView<float> logits = logits_buffer.view({1, vocab_size_});
    runtime::DeviceTensorView<const float> last_hidden(
        current.data + (seq_len - 1) * hidden_dim_, {static_cast<size_t>(1), hidden_dim_});
    lm_head_.forward(context_, last_hidden, logits);

    const int next_token = choose_next_token(runtime::DeviceTensorView<const float>(logits.data, logits.shape));
    token_ids.push_back(next_token);
    if (next_token == tokenizer_.eos_id())
      break;
  }

  if (!token_ids.empty() && token_ids.front() == tokenizer_.bos_id())
    token_ids.erase(token_ids.begin());
  return detokenize(token_ids);
}

vector<int> LlamaModel::tokenize(const std::string& text) const {
  return tokenizer_.encode(text);
}

std::string LlamaModel::detokenize(const vector<int>& token_ids) const {
  return tokenizer_.decode(token_ids);
}

void LlamaModel::validate_model_geometry() const {
  if (hidden_dim_ != kNumQHeads * kHeadDim) {
    throw runtime_error("LlamaModel: hidden dimension does not match hardcoded head geometry");
  }
  if (weights_.embed_tokens().shape.size() != 2 || weights_.lm_head().shape.size() != 2) {
    throw runtime_error("LlamaModel: embedding and lm_head weights must be rank-2");
  }
  if (weights_.lm_head().shape[1] != hidden_dim_) {
    throw runtime_error("LlamaModel: lm_head input dimension must match hidden dimension");
  }

  for (size_t layer_idx = 0; layer_idx < runtime::ModelWeightsGPU::kNumLayers; ++layer_idx) {
    const runtime::LayerWeightsGPU& layer = weights_.layer(layer_idx);
    if (layer.q_proj.meta.shape != vector<size_t>{hidden_dim_, hidden_dim_}) {
      throw runtime_error("LlamaModel: q_proj shape mismatch in layer " +
                          std::to_string(layer_idx));
    }
    if (layer.k_proj.meta.shape != vector<size_t>{kNumKVHeads * kHeadDim, hidden_dim_}) {
      throw runtime_error("LlamaModel: k_proj shape mismatch in layer " +
                          std::to_string(layer_idx));
    }
    if (layer.v_proj.meta.shape != vector<size_t>{kNumKVHeads * kHeadDim, hidden_dim_}) {
      throw runtime_error("LlamaModel: v_proj shape mismatch in layer " +
                          std::to_string(layer_idx));
    }
    if (layer.o_proj.meta.shape != vector<size_t>{hidden_dim_, hidden_dim_}) {
      throw runtime_error("LlamaModel: o_proj shape mismatch in layer " +
                          std::to_string(layer_idx));
    }
    if (layer.gate_proj.meta.shape[1] != hidden_dim_ || layer.up_proj.meta.shape[1] != hidden_dim_ ||
        layer.down_proj.meta.shape[0] != hidden_dim_) {
      throw runtime_error("LlamaModel: FFN input/output dimensions mismatch in layer " +
                          std::to_string(layer_idx));
    }
    if (layer.gate_proj.meta.shape[0] != layer.up_proj.meta.shape[0] ||
        layer.down_proj.meta.shape[1] != layer.gate_proj.meta.shape[0]) {
      throw runtime_error("LlamaModel: FFN hidden dimensions mismatch in layer " +
                          std::to_string(layer_idx));
    }
  }
}

int LlamaModel::choose_next_token(runtime::DeviceTensorView<const float> logits) const {
  next_token_buffer_.resize(logits.shape[0]);
  argmax_.forward(context_, logits, next_token_buffer_.view({logits.shape[0]}));

  int host_token = 0;
  next_token_buffer_.copy_to_host(&host_token, 1, context_.stream());
  context_.synchronize();
  return host_token;
}
