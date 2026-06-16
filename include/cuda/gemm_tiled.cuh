#pragma once

// Tiled shared‑memory GEMM
// C = A * B, where A is M×K, B is K×N, C is M×N
void tiled_gemm(const float* A, const float* B, float* C, int M, int N, int K);