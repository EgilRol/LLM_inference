#include "io/weight_loader.h"

#include <fstream>
#include <limits>
#include <utility>

namespace io {
namespace {

void read_exact(std::ifstream& f, void* dst, size_t bytes, const string& what) {
  f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (!f)
    throw runtime_error("weight_loader: failed to read " + what);
}

}  // namespace

WeightLoader::WeightLoader(string file_path) : index_(std::move(file_path)) {}

const string& WeightLoader::file_path() const { return index_.file_path(); }

bool WeightLoader::has(const string& tensor_name) const {
  return index_.has(tensor_name);
}

const TensorMeta& WeightLoader::meta(const string& tensor_name) const {
  return index_.meta(tensor_name);
}

HostTensor WeightLoader::load_tensor(const string& tensor_name) const {
  const TensorMeta& tensor_meta = meta(tensor_name);
  if (tensor_meta.dtype != TensorDType::FP32) {
    throw runtime_error("weight_loader: HostTensor load only supports FP32 tensors");
  }
  HostTensor out;
  out.meta = tensor_meta;
  // The convenience path materializes one full host tensor, but only one at a time.
  out.data.resize(tensor_meta.num_elements);
  read_tensor(tensor_name, out.data.data(), tensor_meta.num_bytes);
  return out;
}

void WeightLoader::read_tensor(const string& tensor_name, void* dst, size_t bytes) const {
  const TensorMeta& tensor_meta = meta(tensor_name);
  if (bytes != tensor_meta.num_bytes) {
    throw runtime_error("weight_loader: byte count mismatch for tensor '" +
                        tensor_name + "'");
  }
  read_bytes(tensor_name, 0, dst, bytes);
}

void WeightLoader::read_bytes(const string& tensor_name, size_t tensor_offset, void* dst,
                              size_t bytes) const {
  const TensorMeta& tensor_meta = meta(tensor_name);
  if (tensor_offset > tensor_meta.num_bytes || bytes > tensor_meta.num_bytes - tensor_offset) {
    throw runtime_error("weight_loader: read range out of bounds for tensor '" +
                        tensor_name + "'");
  }

  if (tensor_meta.data_offset >
      static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
    throw runtime_error("weight_loader: tensor offset too large for stream seek");
  }
  const size_t absolute_offset = tensor_meta.data_offset + tensor_offset;
  if (absolute_offset > static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
    throw runtime_error("weight_loader: absolute offset too large for stream seek");
  }

  std::ifstream f(file_path(), std::ios::binary);
  if (!f)
    throw runtime_error("weight_loader: cannot open " + file_path());

  // Seek directly to the payload slice using the precomputed file offset.
  f.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
  if (!f)
    throw runtime_error("weight_loader: failed to seek tensor '" + tensor_name + "'");

  read_exact(f, dst, bytes, "tensor payload");
}

}  // namespace io
