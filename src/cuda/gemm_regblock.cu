#include "cuda/gemm_regblock.cuh"

__global__ void gemm_regblock_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    // placeholder: write zero
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = 0.0f; // dummy
    }
}

void gemm_regblock(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm_regblock_kernel<<<grid, block>>>(A, B, C, M, N, K);
}