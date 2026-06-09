#include "cuda/gemm_naive.cuh"

// =============================================================================
// Naive GEMM — one thread per output element
//
// Grid:   ceil(N/32) x ceil(M/32) blocks
// Block:  32 x 32 threads
//
// Thread (row, col) computes:
//
//   C[row][col] = sum over k of A[row][k] * B[k][col]
//
// Memory access pattern:
//   A[row][k]  — same row across k iterations — coalesced across k, 
//                but threads in a warp access same row independently
//   B[k][col]  — different row each iteration — strided, uncoalesced
//   C[row][col] — one write per thread — coalesced across warp
//
// This kernel is memory-bandwidth limited, not compute limited.
// Nsight Compute will confirm high global memory stall cycles.
// =============================================================================

__global__ void naive_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M,
    int N,
    int K
)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}