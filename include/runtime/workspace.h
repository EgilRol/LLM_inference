#pragma once

#include "runtime/device_buffer.h"

namespace runtime {

class Workspace {
 public:
  // Scratch buffers are grown on demand and then reused by name.
  DeviceBuffer<float>& get_buffer(const string& name, size_t count);
  DeviceTensorView<float> get_view(const string& name, const vector<size_t>& shape);

 private:
  unordered_map<string, DeviceBuffer<float>> buffers_;
};

}  // namespace runtime
