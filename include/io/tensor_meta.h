#pragma once

#include "prelude.h"

#include <cstdint>

namespace io {

enum class TensorDType : uint32_t {
  FP32 = 0,
  BF16 = 1,
};

struct TensorMeta {
  // Logical tensor identity from the dump file.
  string name;
  // Row-major extents exactly as stored in the dump header.
  vector<size_t> shape;
  TensorDType dtype = TensorDType::FP32;
  // Precomputed sizes so callers do not redo shape math repeatedly.
  size_t num_elements = 0;
  size_t num_bytes = 0;
  // Absolute byte offset of the tensor payload within the dump file.
  size_t data_offset = 0;
};

size_t num_elements_from_shape(const vector<size_t>& shape);
size_t tensor_dtype_size(TensorDType dtype);

}  // namespace io
