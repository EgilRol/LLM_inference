#pragma once

#include "io/host_tensor.h"
#include "io/weight_index.h"

namespace io {

class WeightLoader {
 public:
  // Can read tensors into HostTensor given a WeightIndex
  explicit WeightLoader(string file_path);

  const string& file_path() const;
  bool has(const string& tensor_name) const;
  const TensorMeta& meta(const string& tensor_name) const;

  HostTensor load_tensor(const string& tensor_name) const;
  // Reads the full tensor payload into caller-owned storage.
  void read_tensor(const string& tensor_name, void* dst, size_t bytes) const;
  // Reads a byte range within one tensor, used by staged GPU upload.
  void read_bytes(const string& tensor_name, size_t tensor_offset, void* dst,
                  size_t bytes) const;

 private:
  WeightIndex index_;
};

}  // namespace io
