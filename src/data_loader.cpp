#include "data_loader.h"
#include "prelude.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
const unsigned char MAGIC[] = {'L', 'L', 'W', 0x01};
}

LoadedTensor load_tensor(const string& file_path, const string& tensor_name) {
  std::ifstream f(file_path, std::ios::binary);
  if (!f)
    throw runtime_error("data_loader: cannot open " + file_path);

  unsigned char magic[4];
  f.read(reinterpret_cast<char*>(magic), 4);
  if (!f || magic[0] != MAGIC[0] || magic[1] != MAGIC[1] || magic[2] != MAGIC[2] ||
      magic[3] != MAGIC[3])
    throw runtime_error("data_loader: invalid magic in " + file_path);

  uint32_t num_tensors;
  f.read(reinterpret_cast<char*>(&num_tensors), 4);
  if (!f)
    throw runtime_error("data_loader: read num_tensors failed");

  for (uint32_t t = 0; t < num_tensors; ++t) {
    uint32_t name_len;
    f.read(reinterpret_cast<char*>(&name_len), 4);
    if (!f)
      throw runtime_error("data_loader: read name_len failed");
    string name(static_cast<size_t>(name_len), '\0');
    f.read(&name[0], name_len);
    if (!f)
      throw runtime_error("data_loader: read name failed");

    uint32_t ndim;
    f.read(reinterpret_cast<char*>(&ndim), 4);
    if (!f)
      throw runtime_error("data_loader: read ndim failed");

    vector<uint32_t> dims(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
      f.read(reinterpret_cast<char*>(&dims[i]), 4);
      if (!f)
        throw runtime_error("data_loader: read shape failed");
    }

    size_t num_floats = 1;
    for (uint32_t d : dims)
      num_floats *= static_cast<size_t>(d);

    if (name == tensor_name) {
      LoadedTensor out;
      out.shape.resize(ndim);
      for (size_t i = 0; i < ndim; ++i)
        out.shape[i] = dims[i];
      out.data.resize(num_floats);
      f.read(reinterpret_cast<char*>(out.data.data()), num_floats * sizeof(float));
      if (!f)
        throw runtime_error("data_loader: read tensor data failed");
      return out;
    }

    // Skip this tensor's data
    const size_t skip_bytes = num_floats * sizeof(float);
    if (skip_bytes > static_cast<size_t>(std::numeric_limits<std::streamoff>::max()))
      throw runtime_error("data_loader: tensor too large to skip");
    f.seekg(static_cast<std::streamoff>(skip_bytes), std::ios::cur);
    if (!f)
      throw runtime_error("data_loader: skip tensor data failed");
  }

  throw runtime_error("data_loader: tensor '" + tensor_name + "' not found in " + file_path);
}
