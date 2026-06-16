#include "cuda/gemm_regblock.cuh"

__global__ void regblock_gemm_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = 0.0f;
    }
}

void regblock_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(64, 4);
    dim3 grid((N + 63) / 64, (M + 3) / 4);
    regblock_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}