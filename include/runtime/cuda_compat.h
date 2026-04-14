#pragma once

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define LLM_RUNTIME_HAS_CUDA 1
#else
using cudaStream_t = void*;
using cudaError_t = int;
#define LLM_RUNTIME_HAS_CUDA 0
#endif
