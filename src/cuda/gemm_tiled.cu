#include "cuda/gemm_tiled.cuh"

__global__ void tiled_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_SIZE = 16;

    // Pad to avoid bank conflicts
    __shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; ++phase) {

        if (row < M && (phase * TILE_SIZE + tx) < K)
            s_A[ty][tx] = A[row * K + (phase * TILE_SIZE + tx)];
        else
            s_A[ty][tx] = 0.0f;

        if (col < N && (phase * TILE_SIZE + ty) < K)
            s_B[ty][tx] = B[(phase * TILE_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += s_A[ty][i] * s_B[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}