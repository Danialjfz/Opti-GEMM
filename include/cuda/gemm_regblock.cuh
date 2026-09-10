#pragma once
__global__ void regblock_gemm(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float*       __restrict__ C,
                           int M, int N, int K);