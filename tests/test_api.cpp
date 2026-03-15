#include "test_api.h"
#include "config.h"
#include "embedding.h"
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
  Embedder emb(EMBED_WEIGHTS_PATH);
  return emb.embed(token_ids);
}

vector<float> TestAPI::matmul(const vector<float> &A, const vector<float> &B,
                              int M, int K, int N) {
  throw runtime_error("Not implemented: you need to implement the "
                      "matmul function here");
}
