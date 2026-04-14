#pragma once

#include "prelude.h"

#include <initializer_list>

namespace runtime {

inline size_t num_elements_from_shape(const vector<size_t>& shape) {
  size_t count = 1;
  for (size_t dim : shape)
    count *= dim;
  return count;
}

template <typename T>
struct DeviceTensorView {
  T* data = nullptr;
  // Shape metadata lives on the host; the device only sees the raw pointer.
  vector<size_t> shape;

  DeviceTensorView() = default;
  DeviceTensorView(T* data_ptr, vector<size_t> tensor_shape)
      : data(data_ptr), shape(std::move(tensor_shape)) {}

  size_t num_elements() const { return num_elements_from_shape(shape); }
};

template <typename T>
DeviceTensorView<T> make_device_tensor_view(T* data, std::initializer_list<size_t> shape) {
  return DeviceTensorView<T>(data, vector<size_t>(shape));
}

}  // namespace runtime
