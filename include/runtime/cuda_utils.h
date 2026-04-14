#pragma once

#include "prelude.h"
#include "runtime/cuda_compat.h"

namespace runtime {

inline void ensure_cuda_available(const string& what) {
#if !LLM_RUNTIME_HAS_CUDA
  throw runtime_error(what + " requires CUDA runtime support");
#else
  (void)what;
#endif
}

inline void check_cuda(cudaError_t status, const string& what) {
#if LLM_RUNTIME_HAS_CUDA
  if (status != cudaSuccess) {
    throw runtime_error(what + " failed: " + string(cudaGetErrorString(status)));
  }
#else
  (void)status;
  throw runtime_error(what + " requires CUDA runtime support");
#endif
}

inline void check_last_launch(const string& what) {
#if LLM_RUNTIME_HAS_CUDA
  check_cuda(cudaGetLastError(), what);
#else
  (void)what;
#endif
}

}  // namespace runtime
