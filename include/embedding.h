#include "prelude.h"

class Embedder {
public:
  explicit Embedder(const string& param_bin_path);
  vector<float> embed(const vector<int>& token_ids);

private:
  vector<float> embedding_matrix;
  size_t hidden_dim_;
  string param_bin_path_;
  void load_embedding_matrix();
};