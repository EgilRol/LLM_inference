#pragma once

#include "runtime/cuda_compat.h"

namespace runtime {

class CudaContext {
 public:
  // Owns the single stream used by the initial synchronous runtime.
  CudaContext();
  ~CudaContext();

  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  cudaStream_t stream() const;
  void synchronize() const;

 private:
  cudaStream_t stream_;
};

}  // namespace runtime
