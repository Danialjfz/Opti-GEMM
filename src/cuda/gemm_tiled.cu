#include "cuda/gemm_tiled.cuh"

__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_SIZE = 16;

    // Pad the second dimension by 1 to avoid shared memory bank conflicts.
    // Shared memory has 32 banks; when TILE_SIZE is a multiple of 32 (like 16
    // on older architectures) successive rows fall into the same banks, causing
    // serialisation. Padding shifts the start of each row, distributing
    // accesses across different banks.
    __shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];   // padding added
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; ++phase) {

        // Load A tile element
        if (row < M && (phase * TILE_SIZE + tx) < K)
            s_A[ty][tx] = A[row * K + (phase * TILE_SIZE + tx)];
        else
            s_A[ty][tx] = 0.0f;

        // Load B tile element
        if (col < N && (phase * TILE_SIZE + ty) < K)
            s_B[ty][tx] = B[(phase * TILE_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        // The compiler can fully unroll this loop because TILE_SIZE is known
        // at compile time, reducing loop overhead.
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += s_A[ty][i] * s_B[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void tiled_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    tiled_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}