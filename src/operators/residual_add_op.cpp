#include "operators/residual_add_op.h"

#include "kernel/kernels.cuh"

namespace {

void validate_residual_tensor(const runtime::DeviceTensorView<const float>& tensor,
                              const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("ResidualAddOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("ResidualAddOp: " + name + " must be rank-2");
}

void validate_residual_tensor(const runtime::DeviceTensorView<float>& tensor,
                              const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("ResidualAddOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("ResidualAddOp: " + name + " must be rank-2");
}

}  // namespace

void ResidualAddOp::forward(const runtime::CudaContext& context,
                            runtime::DeviceTensorView<const float> input,
                            runtime::DeviceTensorView<const float> residual,
                            runtime::DeviceTensorView<float> output) const {
  validate_residual_tensor(input, "input");
  validate_residual_tensor(residual, "residual");
  validate_residual_tensor(output, "output");

  if (input.shape != residual.shape || input.shape != output.shape) {
    throw runtime_error("ResidualAddOp: input, residual, and output shapes must match");
  }

  launch_residual_add(context, input, residual, output);
}

void ResidualAddOp::forward_inplace(const runtime::CudaContext& context,
                                    runtime::DeviceTensorView<float> input_output,
                                    runtime::DeviceTensorView<const float> residual) const {
  validate_residual_tensor(input_output, "input_output");
  validate_residual_tensor(residual, "residual");

  if (input_output.shape != residual.shape)
    throw runtime_error("ResidualAddOp: input_output and residual shapes must match");

  launch_residual_add(
      context, runtime::DeviceTensorView<const float>(input_output.data, input_output.shape),
      residual, input_output);
}
