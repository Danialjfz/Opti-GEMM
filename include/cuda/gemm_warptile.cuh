#pragma once

void warptile_gemm(
    const float* A,
    const float* B,
    float*       C,
    int M,
    int N,
    int K
);