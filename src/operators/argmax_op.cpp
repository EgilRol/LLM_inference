#include "operators/argmax_op.h"

#include "kernel/kernels.cuh"

namespace {

void validate_argmax_tensor(const runtime::DeviceTensorView<const float>& tensor,
                            const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("ArgmaxOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("ArgmaxOp: " + name + " must be rank-2");
}

void validate_argmax_tensor(const runtime::DeviceTensorView<int>& tensor, const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("ArgmaxOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 1)
    throw runtime_error("ArgmaxOp: " + name + " must be rank-1");
}

}  // namespace

void ArgmaxOp::forward(const runtime::CudaContext& context,
                       runtime::DeviceTensorView<const float> input,
                       runtime::DeviceTensorView<int> output) const {
  validate_argmax_tensor(input, "input");
  validate_argmax_tensor(output, "output");

  if (output.shape[0] != input.shape[0])
    throw runtime_error("ArgmaxOp: output length must match input row count");

  launch_argmax(context, input, output);
}
