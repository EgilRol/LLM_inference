#pragma once

#include "io/tensor_meta.h"

namespace io {

struct HostTensor {
  // Metadata and payload travel together for the host-side convenience path.
  TensorMeta meta;
  vector<float> data;
};

}  // namespace io
