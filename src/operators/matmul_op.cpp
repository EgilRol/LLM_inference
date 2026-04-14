#include "operators/matmul_op.h"

#include "kernel/kernels.cuh"

namespace {

void validate_matmul_tensor(const runtime::DeviceTensorView<const float>& tensor,
                            const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("MatmulOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("MatmulOp: " + name + " must be rank-2");
}

void validate_matmul_tensor(const runtime::DeviceTensorView<float>& tensor,
                            const string& name) {
  if (tensor.data == nullptr)
    throw runtime_error("MatmulOp: " + name + " data pointer is null");
  if (tensor.shape.size() != 2)
    throw runtime_error("MatmulOp: " + name + " must be rank-2");
}

}  // namespace

void MatmulOp::forward(const runtime::CudaContext& context,
                       runtime::DeviceTensorView<const float> A,
                       runtime::DeviceTensorView<const float> B,
                       runtime::DeviceTensorView<float> C) const {
  validate_matmul_tensor(A, "A");
  validate_matmul_tensor(B, "B");
  validate_matmul_tensor(C, "C");

  if (A.shape[1] != B.shape[0])
    throw runtime_error("MatmulOp: inner dimensions do not match");
  if (C.shape[0] != A.shape[0] || C.shape[1] != B.shape[1])
    throw runtime_error("MatmulOp: output shape does not match input shapes");

  launch_matmul(context, A, B, C);
}
