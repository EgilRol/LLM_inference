#include "config.h"
#include "data_loader.h"
#include "embedding.h"
#include "prelude.h"

Embedder::Embedder(const string& param_bin_path)
    : hidden_dim_(0), param_bin_path_(param_bin_path) {
  load_embedding_matrix();
}

void Embedder::load_embedding_matrix() {
  LoadedTensor t = load_tensor(param_bin_path_, "model.embed_tokens.weight");
  if (t.shape.size() != 2)
    throw runtime_error("Embedder: expected 2D embedding tensor");
  hidden_dim_ = t.shape[1];
  if (t.shape[1] != static_cast<size_t>(EMBEDDING_DIM))
    throw runtime_error("Embedder: hidden dim (" + std::to_string(t.shape[1]) +
                        ") does not match config (" + std::to_string(EMBEDDING_DIM) + ")");
  embedding_matrix = std::move(t.data);
}

vector<float> Embedder::embed(const vector<int>& token_ids) {
  if (embedding_matrix.empty() || hidden_dim_ == 0)
    return {};
  size_t vocab_size = embedding_matrix.size() / hidden_dim_;
  vector<float> out;
  out.reserve(token_ids.size() * hidden_dim_);
  for (int id : token_ids) {
    if (id < 0 || static_cast<size_t>(id) >= vocab_size)
      continue;
    size_t offset = static_cast<size_t>(id) * hidden_dim_;
    for (size_t j = 0; j < hidden_dim_; ++j)
      out.push_back(embedding_matrix[offset + j]);
  }
  return out;
}
