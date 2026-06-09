#pragma once

// =============================================================================
// Naive CUDA GEMM Kernel
//
// Strategy:   One thread computes one output element C[row][col]
// Memory:     All reads from global memory — no shared memory, no caching
// Weakness:   Uncoalesced B access, redundant global loads, bandwidth-limited
//
// Expected performance: well below cuBLAS — this is the baseline floor
// =============================================================================

__global__ void naive_gemm(
    const float* __restrict__ A,   // [M x K] row-major
    const float* __restrict__ B,   // [K x N] row-major
    float*       __restrict__ C,   // [M x N] row-major
    int M,
    int N,
    int K
);