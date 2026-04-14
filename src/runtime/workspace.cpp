#include "runtime/workspace.h"

namespace runtime {

DeviceBuffer<float>& Workspace::get_buffer(const string& name, size_t count) {
  DeviceBuffer<float>& buffer = buffers_[name];
  if (buffer.size() < count)
    buffer.resize(count);
  return buffer;
}

DeviceTensorView<float> Workspace::get_view(const string& name, const vector<size_t>& shape) {
  return get_buffer(name, num_elements_from_shape(shape)).view(shape);
}

}  // namespace runtime
