#pragma once

#include "prelude.h"

struct LoadedTensor {
  vector<float> data;
  vector<size_t> shape;
};

// Load a tensor by name from a .bin file (dumper format).
// Tensor names match model.safetensors.index.json (e.g. "model.embed_tokens.weight").
// Throws if file cannot be opened, format is invalid, or tensor_name is not found.
LoadedTensor load_tensor(const string& file_path, const string& tensor_name);
