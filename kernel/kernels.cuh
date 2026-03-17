#pragma once

#include "prelude.h"
#include <cuda_runtime.h>

constexpr int TILE_SIZE = 32;

// Matrix multiply (row-major): A[M,K] * B[K,N] -> C[M,N]. Returns flattened C.
vector<float> matmul(const vector<float> &A_h, const vector<float> &B_h, int M,
                     int K, int N);

__global__ void matmul_kernel(float *A, float *B, float *C, int M, int K,
                              int N);
