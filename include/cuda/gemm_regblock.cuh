#pragma once

// Register‑blocked GEMM for high arithmetic intensity
void gemm_regblock(const float* A, const float* B, float* C, int M, int N, int K);