#include "runtime/rope_tables.h"

#include <cmath>

namespace runtime {

RoPETables::RoPETables(const CudaContext& context, size_t max_positions, size_t head_dim,
                       float base)
    : max_positions_(max_positions),
      head_dim_(head_dim),
      base_(base),
      cos_(max_positions * (head_dim / 2)),
      sin_(max_positions * (head_dim / 2)) {
  if (max_positions_ == 0)
    throw runtime_error("RoPETables: max_positions must be positive");
  if (head_dim_ == 0 || head_dim_ % 2 != 0)
    throw runtime_error("RoPETables: head_dim must be positive and even");
  if (base_ <= 0.0f)
    throw runtime_error("RoPETables: base must be positive");

  const size_t pair_dim = head_dim_ / 2;
  vector<float> host_cos(max_positions_ * pair_dim);
  vector<float> host_sin(max_positions_ * pair_dim);

  for (size_t pos = 0; pos < max_positions_; ++pos) {
    for (size_t pair = 0; pair < pair_dim; ++pair) {
      const float theta =
          std::pow(base_, -static_cast<float>(pair) / static_cast<float>(pair_dim));
      const float angle = static_cast<float>(pos) * theta;
      host_cos[pos * pair_dim + pair] = std::cos(angle);
      host_sin[pos * pair_dim + pair] = std::sin(angle);
    }
  }

  cos_.copy_from_host(host_cos.data(), host_cos.size(), context.stream());
  sin_.copy_from_host(host_sin.data(), host_sin.size(), context.stream());
  context.synchronize();
}

DeviceTensorView<const float> RoPETables::cos() const {
  return cos_.view({max_positions_, head_dim_ / 2});
}

DeviceTensorView<const float> RoPETables::sin() const {
  return sin_.view({max_positions_, head_dim_ / 2});
}

size_t RoPETables::max_positions() const { return max_positions_; }

size_t RoPETables::head_dim() const { return head_dim_; }

size_t RoPETables::pair_dim() const { return head_dim_ / 2; }

float RoPETables::base() const { return base_; }

}  // namespace runtime
