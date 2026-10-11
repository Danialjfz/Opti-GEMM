#pragma once

// =============================================================================
// Register-blocked GEMM
//
//   Block tile   : 64 x 64 output elements
//   Thread block : 16 x 16 = 256 threads (8 full warps)
//   Micro-tile   : 4 x 4 output elements per thread, accumulated in registers
//
// Host-side launcher: owns the grid/block configuration so callers never
// pick launch dims and a misconfigured launch is impossible.
// =============================================================================
void regblock_gemm(const float* A,
                   const float* B,
                   float*       C,
                   int M, int N, int K);
