#pragma once

#include "prelude.h"

#include <cstdint>

enum class ReferenceFixtureDType : uint32_t {
  Int32 = 1,
  Float32 = 2,
};

struct ReferenceFixtureTensor {
  string name;
  ReferenceFixtureDType dtype;
  vector<size_t> shape;
  vector<int32_t> i32_data;
  vector<float> f32_data;

  size_t num_elements() const;
};

class ReferenceFixture {
 public:
  explicit ReferenceFixture(string file_path);

  const string& file_path() const;
  bool has(const string& tensor_name) const;
  const ReferenceFixtureTensor& tensor(const string& tensor_name) const;
  int32_t scalar_i32(const string& tensor_name) const;

 private:
  string file_path_;
  unordered_map<string, ReferenceFixtureTensor> tensors_;
};
