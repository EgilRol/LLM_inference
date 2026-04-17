#include "operators/swiglu_op.h"

#include "kernel/kernels.cuh"

namespace {

void validate_swiglu_tensor(const runtime::DeviceTensorView<const float>& tensor,
                            const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("SwiGLUOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("SwiGLUOp: " + name + " must be rank-2");
}

void validate_swiglu_tensor(const runtime::DeviceTensorView<float>& tensor, const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("SwiGLUOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("SwiGLUOp: " + name + " must be rank-2");
}

}  // namespace

void SwiGLUOp::forward(const runtime::CudaContext& context,
                       runtime::DeviceTensorView<const float> gate,
                       runtime::DeviceTensorView<const float> up,
                       runtime::DeviceTensorView<float> output) const {
  validate_swiglu_tensor(gate, "gate");
  validate_swiglu_tensor(up, "up");
  validate_swiglu_tensor(output, "output");

  if (gate.shape != up.shape || gate.shape != output.shape)
    throw runtime_error("SwiGLUOp: gate, up, and output shapes must match");

  launch_swiglu(context, gate, up, output);
}
