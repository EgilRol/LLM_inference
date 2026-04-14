#pragma once

#include "io/tensor_meta.h"

namespace io {

class WeightIndex {
 public:
  // Creates the entries<tensor_name, TensorMeta> map for a given binary file_path
  explicit WeightIndex(string file_path);

  const string& file_path() const;
  bool has(const string& tensor_name) const;
  const TensorMeta& meta(const string& tensor_name) const;
  const unordered_map<string, TensorMeta>& entries() const;

 private:
  string file_path_;
  unordered_map<string, TensorMeta> entries_;
};

}  // namespace io
