#pragma once

// Warp‑tile GEMM using warp‑level matrix operations
void warptile_gemm(const float* A, const float* B, float* C, int M, int N, int K);