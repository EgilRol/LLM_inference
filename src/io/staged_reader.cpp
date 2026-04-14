#include "io/staged_reader.h"

#include <algorithm>

namespace io {
namespace {

#if LLM_RUNTIME_HAS_CUDA
void check_cuda(cudaError_t status, const string& what) {
  if (status != cudaSuccess) {
    throw runtime_error("staged_reader: " + what + " failed: " +
                        string(cudaGetErrorString(status)));
  }
}
#endif

}  // namespace

StagedReader::StagedReader(size_t staging_bytes)
    : staging_buffer_(nullptr), staging_bytes_(staging_bytes) {
  if (staging_bytes_ == 0)
    throw runtime_error("staged_reader: staging buffer size must be greater than zero");
#if LLM_RUNTIME_HAS_CUDA
  // Pinned host memory gives the startup upload path a single reusable staging area.
  check_cuda(cudaMallocHost(&staging_buffer_, staging_bytes_),
             "cudaMallocHost staging buffer");
#endif
}

StagedReader::~StagedReader() {
#if LLM_RUNTIME_HAS_CUDA
  if (staging_buffer_ != nullptr)
    cudaFreeHost(staging_buffer_);
#endif
}

size_t StagedReader::staging_bytes() const { return staging_bytes_; }

void StagedReader::upload_tensor(const WeightLoader& loader, const string& tensor_name,
                                 void* device_ptr, cudaStream_t stream) const {
#if !LLM_RUNTIME_HAS_CUDA
  (void)loader;
  (void)tensor_name;
  (void)device_ptr;
  (void)stream;
  throw runtime_error("staged_reader: CUDA runtime headers are not available");
#else
  const TensorMeta& tensor_meta = loader.meta(tensor_name);
  size_t uploaded = 0;
  while (uploaded < tensor_meta.num_bytes) {
    // Large tensors can be streamed in fixed-size chunks to avoid large host buffers.
    const size_t chunk_bytes = std::min(staging_bytes_, tensor_meta.num_bytes - uploaded);
    loader.read_bytes(tensor_name, uploaded, staging_buffer_, chunk_bytes);

    void* device_chunk_ptr = static_cast<void*>(static_cast<char*>(device_ptr) + uploaded);
    if (stream == nullptr) {
      check_cuda(cudaMemcpy(device_chunk_ptr, staging_buffer_, chunk_bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy host to device");
    } else {
      check_cuda(cudaMemcpyAsync(device_chunk_ptr, staging_buffer_, chunk_bytes,
                                 cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync host to device");
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }
    uploaded += chunk_bytes;
  }
#endif
}

}  // namespace io
