#include "reference_fixture.h"

#include <fstream>
#include <limits>

namespace {

constexpr unsigned char kMagic[] = {'R', 'F', 'X', 0x01};

void read_exact(std::ifstream& f, void* dst, size_t bytes, const string& what) {
  f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (!f)
    throw runtime_error("reference_fixture: failed to read " + what);
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
    throw runtime_error("reference_fixture: overflow while computing " + what);
  return a * b;
}

size_t num_elements_from_shape(const vector<size_t>& shape) {
  size_t count = 1;
  for (size_t dim : shape)
    count = checked_multiply(count, dim, "tensor element count");
  return count;
}

size_t dtype_size(ReferenceFixtureDType dtype) {
  switch (dtype) {
    case ReferenceFixtureDType::Int32:
      return sizeof(int32_t);
    case ReferenceFixtureDType::Float32:
      return sizeof(float);
  }
  throw runtime_error("reference_fixture: unsupported dtype");
}

ReferenceFixtureDType read_dtype(std::ifstream& f) {
  const uint32_t raw_dtype = read_u32(f, "dtype");
  switch (raw_dtype) {
    case static_cast<uint32_t>(ReferenceFixtureDType::Int32):
      return ReferenceFixtureDType::Int32;
    case static_cast<uint32_t>(ReferenceFixtureDType::Float32):
      return ReferenceFixtureDType::Float32;
    default:
      throw runtime_error("reference_fixture: unsupported dtype " + std::to_string(raw_dtype));
  }
}

}  // namespace

size_t ReferenceFixtureTensor::num_elements() const {
  return num_elements_from_shape(shape);
}

ReferenceFixture::ReferenceFixture(string file_path) : file_path_(std::move(file_path)) {
  std::ifstream f(file_path_, std::ios::binary);
  if (!f)
    throw runtime_error("reference_fixture: cannot open " + file_path_);

  unsigned char magic[4];
  read_exact(f, magic, sizeof(magic), "magic");
  if (magic[0] != kMagic[0] || magic[1] != kMagic[1] || magic[2] != kMagic[2] ||
      magic[3] != kMagic[3]) {
    throw runtime_error("reference_fixture: invalid magic in " + file_path_);
  }

  const uint32_t num_tensors = read_u32(f, "num_tensors");
  vector<ReferenceFixtureTensor> ordered_tensors;
  ordered_tensors.reserve(num_tensors);

  for (uint32_t i = 0; i < num_tensors; ++i) {
    const uint32_t name_len = read_u32(f, "name_len");
    string name(static_cast<size_t>(name_len), '\0');
    read_exact(f, name.data(), static_cast<size_t>(name_len), "tensor name");
    const ReferenceFixtureDType dtype = read_dtype(f);

    const uint32_t ndim = read_u32(f, "ndim");
    vector<size_t> shape(static_cast<size_t>(ndim));
    for (uint32_t dim_idx = 0; dim_idx < ndim; ++dim_idx)
      shape[dim_idx] = static_cast<size_t>(read_u32(f, "shape dim"));

    ReferenceFixtureTensor tensor;
    tensor.name = std::move(name);
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    ordered_tensors.push_back(std::move(tensor));
  }

  for (ReferenceFixtureTensor& tensor : ordered_tensors) {
    const size_t payload_bytes =
        checked_multiply(tensor.num_elements(), dtype_size(tensor.dtype), "tensor payload bytes");
    if (tensor.dtype == ReferenceFixtureDType::Int32) {
      tensor.i32_data.resize(tensor.num_elements());
      read_exact(f, tensor.i32_data.data(), payload_bytes, "int32 payload");
    } else {
      tensor.f32_data.resize(tensor.num_elements());
      read_exact(f, tensor.f32_data.data(), payload_bytes, "float32 payload");
    }

    if (tensors_.count(tensor.name) != 0)
      throw runtime_error("reference_fixture: duplicate tensor '" + tensor.name + "'");
    tensors_.emplace(tensor.name, std::move(tensor));
  }
}

const string& ReferenceFixture::file_path() const { return file_path_; }

bool ReferenceFixture::has(const string& tensor_name) const {
  return tensors_.find(tensor_name) != tensors_.end();
}

const ReferenceFixtureTensor& ReferenceFixture::tensor(const string& tensor_name) const {
  auto it = tensors_.find(tensor_name);
  if (it == tensors_.end()) {
    throw runtime_error("reference_fixture: tensor '" + tensor_name + "' not found in " +
                        file_path_);
  }
  return it->second;
}

int32_t ReferenceFixture::scalar_i32(const string& tensor_name) const {
  const ReferenceFixtureTensor& entry = tensor(tensor_name);
  if (entry.dtype != ReferenceFixtureDType::Int32)
    throw runtime_error("reference_fixture: tensor '" + tensor_name + "' is not int32");
  if (entry.i32_data.size() != 1)
    throw runtime_error("reference_fixture: tensor '" + tensor_name + "' is not a scalar");
  return entry.i32_data[0];
}
