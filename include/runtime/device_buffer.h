#pragma once

#include "runtime/cuda_compat.h"
#include "runtime/cuda_utils.h"
#include "runtime/device_tensor.h"

#include <utility>

namespace runtime {

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() : data_(nullptr), count_(0) {}
  explicit DeviceBuffer(size_t count) : data_(nullptr), count_(0) { resize(count); }

  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void resize(size_t count) {
    if (count == count_)
      return;
    reset();
    if (count == 0)
      return;
    // DeviceBuffer is the owning RAII wrapper for one cudaMalloc allocation.
    ensure_cuda_available("DeviceBuffer::resize");
#if LLM_RUNTIME_HAS_CUDA
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
               "cudaMalloc device buffer");
#endif
    count_ = count;
  }

  void reset() {
#if LLM_RUNTIME_HAS_CUDA
    if (data_ != nullptr)
      cudaFree(data_);
#endif
    data_ = nullptr;
    count_ = 0;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }

  void copy_from_host(const T* src, size_t count, cudaStream_t stream = nullptr) {
    if (count > count_)
      throw runtime_error("DeviceBuffer::copy_from_host exceeds allocation");
    if (count == 0)
      return;
    ensure_cuda_available("DeviceBuffer::copy_from_host");
#if LLM_RUNTIME_HAS_CUDA
    if (stream == nullptr) {
      check_cuda(cudaMemcpy(data_, src, count * sizeof(T), cudaMemcpyHostToDevice),
                 "cudaMemcpy host to device");
    } else {
      check_cuda(cudaMemcpyAsync(data_, src, count * sizeof(T), cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync host to device");
    }
#else
    (void)src;
    (void)stream;
#endif
  }

  void copy_to_host(T* dst, size_t count, cudaStream_t stream = nullptr) const {
    if (count > count_)
      throw runtime_error("DeviceBuffer::copy_to_host exceeds allocation");
    if (count == 0)
      return;
    ensure_cuda_available("DeviceBuffer::copy_to_host");
#if LLM_RUNTIME_HAS_CUDA
    if (stream == nullptr) {
      check_cuda(cudaMemcpy(dst, data_, count * sizeof(T), cudaMemcpyDeviceToHost),
                 "cudaMemcpy device to host");
    } else {
      check_cuda(cudaMemcpyAsync(dst, data_, count * sizeof(T), cudaMemcpyDeviceToHost, stream),
                 "cudaMemcpyAsync device to host");
    }
#else
    (void)dst;
    (void)stream;
#endif
  }

  DeviceTensorView<T> view(const vector<size_t>& shape) {
    if (num_elements_from_shape(shape) > count_)
      throw runtime_error("DeviceBuffer::view shape exceeds allocation");
    return DeviceTensorView<T>(data_, shape);
  }

  DeviceTensorView<const T> view(const vector<size_t>& shape) const {
    if (num_elements_from_shape(shape) > count_)
      throw runtime_error("DeviceBuffer::view shape exceeds allocation");
    return DeviceTensorView<const T>(data_, shape);
  }

 private:
  T* data_;
  size_t count_;
};

}  // namespace runtime
