#include "operators/rmsnorm_op.h"

#include "kernel/kernels.cuh"

namespace {

void validate_rmsnorm_tensor(const runtime::DeviceTensorView<const float>& tensor,
                             const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("RmsNormOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("RmsNormOp: " + name + " must be rank-2");
}

void validate_rmsnorm_tensor(const runtime::DeviceTensorView<float>& tensor,
                             const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("RmsNormOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("RmsNormOp: " + name + " must be rank-2");
}

}  // namespace

RmsNormOp::RmsNormOp(runtime::DeviceTensorView<const __nv_bfloat16> gamma, float epsilon)
    : gamma_(std::move(gamma)), epsilon_(epsilon) {
  if (gamma_.data == nullptr)
    throw runtime_error("RmsNormOp: gamma data pointer is null");
  if (gamma_.shape.size() != 1)
    throw runtime_error("RmsNormOp: gamma must be rank-1");
}

void RmsNormOp::forward(const runtime::CudaContext& context,
                        runtime::DeviceTensorView<const float> input,
                        runtime::DeviceTensorView<float> output) const {
  validate_rmsnorm_tensor(input, "input");
  validate_rmsnorm_tensor(output, "output");

  if (input.shape != output.shape)
    throw runtime_error("RmsNormOp: input and output shapes must match");
  if (gamma_.shape[0] != input.shape[1])
    throw runtime_error("RmsNormOp: gamma length must match hidden dimension");

  launch_rmsnorm(context, input, gamma_, output, epsilon_);
}

runtime::DeviceTensorView<const __nv_bfloat16> RmsNormOp::gamma() const { return gamma_; }

float RmsNormOp::epsilon() const { return epsilon_; }
