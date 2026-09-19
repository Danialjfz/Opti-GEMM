#include "cuda/gemm_regblock.cuh"

// Register-blocked GEMM kernel using a small shared-memory tiling strategy.
// The idea is to load a tile of A and B into shared memory, then accumulate
// the partial dot products for each thread block before writing the final values.
__global__ void regblock_gemm_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
    // Tile dimensions for the shared-memory staging area.
    constexpr int TILE_SIZE = 16;
    constexpr int THREAD_TILE = 4;

    // Shared-memory tiles hold the current A/B blocks being processed.
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Global matrix indices for this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int smem_row = tid / TILE_SIZE;
    int smem_col = tid % TILE_SIZE;
    
    int global_row = blockIdx.y * TILE_SIZE + smem_row;
    int global_k = blockIdx.x * TILE_SIZE + smem_col;

    // Per-thread accumulator for the local output tile.
    float acc[THREAD_TILE][THREAD_TILE] = {0.0f};

    // Iterate over K in tiles to keep the working set within shared memory.
    for (int phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; ++phase) {

        global_row = blockIdx.y * TILE_SIZE + smem_row; 
        global_k = phase * TILE_SIZE + smem_col;

        // Load the A tile for this phase with bounds checking.
        if (global_row < M && global_k < K){
        As[smem_row][smem_col] = A[global_row * K + global_k];
        } else{
            As[smem_row][smem_col] = 0.0f;
        }
        global_k = phase * TILE_SIZE + smem_row; 
        int global_col = blockIdx.x * TILE_SIZE + smem_col;
        if (global_k < K && global_col < N) { 
            Bs[smem_row][smem_col] = B[global_k * N + global_col]; } 
        else { 
            Bs[smem_row][smem_col] = 0.0f; }
        // syncronize to ensure all threads have loaded their data into shared memory.
        

    }
    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {

        // Keep A values in registers.
        float a[THREAD_TILE];

        // Keep B values in registers.
        float b[THREAD_TILE];

        for (int i = 0; i < THREAD_TILE; ++i) {
            a[i] = As[thread_row + i][k];
        }

        for (int j = 0; j < THREAD_TILE; ++j) {
            b[j] = Bs[k][thread_col + j];
        }

        // Compute the 4x4 C micro-tile.
        for (int i = 0; i < THREAD_TILE; ++i) {
            for (int j = 0; j < THREAD_TILE; ++j) {

                acc[i][j] += a[i] * b[j];

            }
        }
    }
}

