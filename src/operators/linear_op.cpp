#include "operators/linear_op.h"

#include "kernel/kernels.cuh"

LinearOp::LinearOp(runtime::DeviceTensorView<const __nv_bfloat16> weight)
    : weight_(std::move(weight)) {
  if (weight_.data == nullptr)
    throw runtime_error("LinearOp: weight data pointer is null");
  if (weight_.shape.size() != 2)
    throw runtime_error("LinearOp: weight must be rank-2");
}

void LinearOp::forward(const runtime::CudaContext& context,
                       runtime::DeviceTensorView<const float> input,
                       runtime::DeviceTensorView<float> output) const {
  if (input.data == nullptr)
    throw runtime_error("LinearOp: input data pointer is null");
  if (output.data == nullptr)
    throw runtime_error("LinearOp: output data pointer is null");
  if (input.shape.size() != 2 || output.shape.size() != 2)
    throw runtime_error("LinearOp: input and output must be rank-2");
  if (input.shape[1] != in_dim())
    throw runtime_error("LinearOp: input hidden dimension does not match weight");
  if (output.shape[0] != input.shape[0] || output.shape[1] != out_dim())
    throw runtime_error("LinearOp: output shape does not match linear projection");

  launch_linear_matmul(context, input, weight_, output);
}

runtime::DeviceTensorView<const __nv_bfloat16> LinearOp::weight() const { return weight_; }

size_t LinearOp::in_dim() const { return weight_.shape[1]; }

size_t LinearOp::out_dim() const { return weight_.shape[0]; }
