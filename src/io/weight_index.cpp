#include "io/weight_index.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace io {
namespace {

constexpr unsigned char kMagic[] = {'L', 'L', 'W', 0x01};

void read_exact(std::ifstream& f, void* dst, size_t bytes, const string& what) {
  f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (!f)
    throw runtime_error("weight_index: failed to read " + what);
}

uint32_t read_u32(std::ifstream& f, const string& what) {
  uint32_t value = 0;
  read_exact(f, &value, sizeof(value), what);
  return value;
}

size_t checked_multiply(size_t a, size_t b, const string& what) {
  if (a == 0 || b == 0)
    return 0;
  if (a > std::numeric_limits<size_t>::max() / b)
    throw runtime_error("weight_index: overflow while computing " + what);
  return a * b;
}

}  // namespace

size_t num_elements_from_shape(const vector<size_t>& shape) {
  size_t num_elements = 1;
  for (size_t dim : shape)
    num_elements = checked_multiply(num_elements, dim, "tensor element count");
  return num_elements;
}

WeightIndex::WeightIndex(string file_path) : file_path_(std::move(file_path)) {
  std::ifstream f(file_path_, std::ios::binary);
  if (!f)
    throw runtime_error("weight_index: cannot open " + file_path_);

  unsigned char magic[4];
  read_exact(f, magic, sizeof(magic), "magic");
  if (magic[0] != kMagic[0] || magic[1] != kMagic[1] || magic[2] != kMagic[2] ||
      magic[3] != kMagic[3]) {
    throw runtime_error("weight_index: invalid magic in " + file_path_);
  }

  const uint32_t num_tensors = read_u32(f, "num_tensors");

  vector<TensorMeta> ordered_entries;
  ordered_entries.reserve(num_tensors);
  for (uint32_t i = 0; i < num_tensors; ++i) {
    // The dump header stores metadata for every tensor before any raw payloads.
    const uint32_t name_len = read_u32(f, "name_len");
    string name(static_cast<size_t>(name_len), '\0');
    read_exact(f, name.data(), static_cast<size_t>(name_len), "tensor name");

    const uint32_t ndim = read_u32(f, "ndim");
    vector<size_t> shape(static_cast<size_t>(ndim));
    for (uint32_t dim_idx = 0; dim_idx < ndim; ++dim_idx) {
      shape[dim_idx] = static_cast<size_t>(read_u32(f, "shape dim"));
    }

    TensorMeta meta;
    meta.name = std::move(name);
    meta.shape = std::move(shape);
    meta.dtype = TensorDType::FP32;
    meta.num_elements = num_elements_from_shape(meta.shape);
    meta.num_bytes =
        checked_multiply(meta.num_elements, sizeof(float), "tensor byte size");
    ordered_entries.push_back(std::move(meta));
  }

  const std::streamoff header_end = f.tellg();
  if (header_end < 0)
    throw runtime_error("weight_index: failed to compute data offset");

  f.seekg(0, std::ios::end);
  const std::streamoff file_size = f.tellg();
  if (file_size < 0)
    throw runtime_error("weight_index: failed to determine file size");

  size_t data_offset = static_cast<size_t>(header_end);
  const size_t total_size = static_cast<size_t>(file_size);

  for (TensorMeta& meta : ordered_entries) {
    if (entries_.count(meta.name) != 0)
      throw runtime_error("weight_index: duplicate tensor '" + meta.name + "'");
    // Payloads are packed immediately after the header in the same tensor order.
    meta.data_offset = data_offset;
    const size_t tensor_end = checked_multiply(1, meta.num_bytes, "tensor end");
    if (meta.data_offset > total_size || tensor_end > total_size - meta.data_offset) {
      throw runtime_error("weight_index: tensor '" + meta.name +
                          "' extends past end of file");
    }
    data_offset += meta.num_bytes;
    entries_.emplace(meta.name, meta);
  }
}

const string& WeightIndex::file_path() const { return file_path_; }

bool WeightIndex::has(const string& tensor_name) const {
  return entries_.find(tensor_name) != entries_.end();
}

const TensorMeta& WeightIndex::meta(const string& tensor_name) const {
  auto it = entries_.find(tensor_name);
  if (it == entries_.end())
    throw runtime_error("weight_index: tensor '" + tensor_name + "' not found in " +
                        file_path_);
  return it->second;
}

const unordered_map<string, TensorMeta>& WeightIndex::entries() const {
  return entries_;
}

}  // namespace io
