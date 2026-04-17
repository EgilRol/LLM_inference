#include "config.h"
#include "model/llama_model.h"
#include "prelude.h"

#include <cstdlib>

int main(int argc, char** argv) {
  const string prompt = argc > 1 ? argv[1] : "Hello world";
  const size_t max_new_tokens =
      argc > 2 ? static_cast<size_t>(std::strtoul(argv[2], nullptr, 10)) : 16;

  LlamaModel model(TOKENIZER_PATH, WEIGHTS_DIR_PATH);
  cout << model.generate(prompt, max_new_tokens) << "\n";
  return 0;
}
