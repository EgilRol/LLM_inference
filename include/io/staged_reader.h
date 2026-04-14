#pragma once

#include "io/weight_loader.h"
#include "runtime/cuda_compat.h"

namespace io {

class StagedReader {
 public:
  // Reuses one pinned host buffer while streaming tensors into GPU memory.
  explicit StagedReader(size_t staging_bytes);
  ~StagedReader();

  // Disable copying
  StagedReader(const StagedReader&) = delete;
  StagedReader& operator=(const StagedReader&) = delete;

  size_t staging_bytes() const;
  // Streams one tensor from disk into an already-allocated device buffer.
  void upload_tensor(const WeightLoader& loader, const string& tensor_name,
                     void* device_ptr, cudaStream_t stream = nullptr) const;

 private:
  void* staging_buffer_;
  size_t staging_bytes_;
};

}  // namespace io
