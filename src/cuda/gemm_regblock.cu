#include "cuda/gemm_tiled.cuh"

__global__ void tiled_gemm_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    // placeholder: write zero
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = 0.0f; // dummy
    }
}

void tiled_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    tiled_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}