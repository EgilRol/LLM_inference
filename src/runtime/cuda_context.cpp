#include "runtime/cuda_context.h"

#include "runtime/cuda_utils.h"

namespace runtime {

CudaContext::CudaContext() : stream_(nullptr) {
#if LLM_RUNTIME_HAS_CUDA
  check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
#endif
}

CudaContext::~CudaContext() {
#if LLM_RUNTIME_HAS_CUDA
  if (stream_ != nullptr)
    cudaStreamDestroy(stream_);
#endif
}

cudaStream_t CudaContext::stream() const { return stream_; }

void CudaContext::synchronize() const {
  ensure_cuda_available("CudaContext::synchronize");
#if LLM_RUNTIME_HAS_CUDA
  check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
#endif
}

}  // namespace runtime
