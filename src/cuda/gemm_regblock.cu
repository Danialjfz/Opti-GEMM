#include "cuda/gemm_regblock.cuh"

__global__ void regblock_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    constexpr int TILE_SIZE  = 16;
    constexpr int THREAD_TILE = 4;

    // 16x16 shared-memory tiles.
    // +1 padding helps reduce shared-memory bank conflicts.
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    /*
     * We have:
     *
     *   16 x 16 output tile
     *
     * and each thread computes:
     *
     *   4 x 4 output elements
     *
     * Therefore:
     *
     *   16 / 4 = 4 threads in X
     *   16 / 4 = 4 threads in Y
     *
     * so each block contains only 4 x 4 = 16 threads.
     */
    constexpr int THREADS_PER_DIM = TILE_SIZE / THREAD_TILE;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Which 4x4 output micro-tile belongs to this thread?
    int thread_row = ty * THREAD_TILE;
    int thread_col = tx * THREAD_TILE;

    // Global starting position of this block's 16x16 C tile.
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;

    // Each thread computes a 4x4 C tile.
    float acc[THREAD_TILE][THREAD_TILE] = {0.0f};

    /*
     * Loop over the K dimension.
     *
     * For every phase:
     *
     *   A: 16x16 tile
     *   B: 16x16 tile
     *
     * is loaded into shared memory.
     */
    for (int phase = 0;
         phase < (K + TILE_SIZE - 1) / TILE_SIZE;
         ++phase)
    {
        /*
         * ---------------------------------------------------------
         * Load A and B tiles into shared memory
         * ---------------------------------------------------------
         *
         * There are only 16 threads, but each tile has 256 elements.
         * Therefore each thread loads multiple elements.
         *
         * A thread loads a 4x4 region of A and a 4x4 region of B.
         */
        for (int i = 0; i < THREAD_TILE; ++i) {
            for (int j = 0; j < THREAD_TILE; ++j) {

                // Position inside shared-memory tile.
                int smem_row = thread_row + i;
                int smem_col = thread_col + j;

                // Global position for A.
                int A_row = block_row + smem_row;
                int A_col = phase * TILE_SIZE + smem_col;

                if (A_row < M && A_col < K) {
                    As[smem_row][smem_col] =
                        A[A_row * K + A_col];
                } else {
                    As[smem_row][smem_col] = 0.0f;
                }

                // Global position for B.
                int B_row = phase * TILE_SIZE + smem_row;
                int B_col = block_col + smem_col;

                if (B_row < K && B_col < N) {
                    Bs[smem_row][smem_col] =
                        B[B_row * N + B_col];
                } else {
                    Bs[smem_row][smem_col] = 0.0f;
                }
            }
        }

        /*
         * IMPORTANT:
         *
         * Every thread must finish loading before any thread
         * starts reading As/Bs.
         */
        __syncthreads();

        /*
         * ---------------------------------------------------------
         * Compute this phase
         * ---------------------------------------------------------
         *
         * For every k:
         *
         *   A contributes 4 values:
         *
         *       A[thread_row + 0][k]
         *       A[thread_row + 1][k]
         *       A[thread_row + 2][k]
         *       A[thread_row + 3][k]
         *
         *   B contributes 4 values:
         *
         *       B[k][thread_col + 0]
         *       B[k][thread_col + 1]
         *       B[k][thread_col + 2]
         *       B[k][thread_col + 3]
         *
         * These produce 16 multiply-adds for the 4x4 C tile.
         */
        for (int k = 0; k < TILE_SIZE; ++k) {

            // Load A values into registers.
            float a[THREAD_TILE];

            for (int i = 0; i < THREAD_TILE; ++i) {
                a[i] = As[thread_row + i][k];
            }

            // Load B values into registers.
            float b[THREAD_TILE];

            for (int j = 0; j < THREAD_TILE; ++j) {
                b[j] = Bs[k][thread_col + j];
            }

            // 4x4 register-blocked computation.
            for (int i = 0; i < THREAD_TILE; ++i) {
                for (int j = 0; j < THREAD_TILE; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        /*
         * IMPORTANT:
         *
         * We are about to overwrite As and Bs in the next phase.
         * Therefore every thread must finish reading them first.
         */
        __syncthreads();
    }

    /*
     * ---------------------------------------------------------
     * Store the 4x4 register tile into C
     * ---------------------------------------------------------
     */
    for (int i = 0; i < THREAD_TILE; ++i) {
        for (int j = 0; j < THREAD_TILE; ++j) {

            int row = block_row + thread_row + i;
            int col = block_col + thread_col + j;

            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}