#include "cuda/gemm_regblock.cuh"

// =============================================================================
// Register-blocked GEMM
//
//   Block tile   : 64 x 64 output elements
//   Thread block : 16 x 16 = 256 threads (8 full warps)
//   Micro-tile   : 4 x 4 output elements per thread, accumulated in registers
//
// Why this configuration
// ----------------------
//
// The first revision of this kernel used a 16x16 block tile with a 4x4
// thread block — only 16 threads. That has two structural problems:
//
//   1. Half-warp waste. A warp has 32 lanes; a 16-thread block runs as a
//      warp with half its lanes idle, so half of every issue slot is lost.
//
//   2. Occupancy cap. GPUs cap the number of resident *blocks* per SM
//      (e.g. 16 on Turing). 16 blocks x 16 threads = 256 threads/SM, a
//      25% occupancy ceiling no matter how small the kernel is.
//
// Scaling the block tile to 64x64 with a 16x16 thread block fixes both:
// 256 threads = 8 full warps, and the resident-block limit now allows
// thousands of threads per SM. Shared memory becomes the limiter instead
// (2 x 64 x 65 x 4 B = ~32.5 KB per block), which is the normal regime
// for high-performance GEMM: high arithmetic intensity per thread covers
// latency better than raw thread count.
//
// Arithmetic intensity per block tile:
//
//   FLOPs        = 2 * 64 * 64 * K
//   Global bytes = 4 * (64*K + 64*K)
//   ratio        = 16 FLOP/byte   (vs 4 FLOP/byte for the 16x16 tile)
// =============================================================================

__global__ void regblock_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M,
    int N,
    int K
) {
    constexpr int TILE_SIZE       = 64;  // block tile: 64 x 64 outputs
    constexpr int THREAD_TILE     = 4;   // micro-tile: 4 x 4 outputs/thread
    constexpr int THREADS_PER_DIM = TILE_SIZE / THREAD_TILE;  // 16

    // 64x64 shared-memory tiles.
    // +1 padding keeps column reads conflict-free: with stride 65,
    // consecutive rows land on consecutive banks.
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Which 4x4 output micro-tile belongs to this thread?
    const int thread_row = ty * THREAD_TILE;
    const int thread_col = tx * THREAD_TILE;

    // Global starting position of this block's 64x64 C tile.
    const int block_row = blockIdx.y * TILE_SIZE;
    const int block_col = blockIdx.x * TILE_SIZE;

    // Each thread accumulates its 4x4 C micro-tile in registers.
    float acc[THREAD_TILE][THREAD_TILE] = {0.0f};

    /*
     * Loop over the K dimension in 64-wide phases.
     *
     * Every phase cooperatively loads a 64x64 tile of A and a 64x64 tile
     * of B into shared memory, then consumes it from registers.
     *
     * Each tile holds 4096 elements; with 256 threads, every thread loads
     * 16 elements (a 4x4 region) of A and of B.
     */
    for (int phase = 0;
         phase < (K + TILE_SIZE - 1) / TILE_SIZE;
         ++phase)
    {
        // -----------------------------------------------------------------
        // Load A and B tiles into shared memory
        //
        // Thread (tx, ty) loads the 4x4 region starting at
        // (thread_row, thread_col) of each tile — the same region it will
        // later read from, which keeps the mapping easy to reason about.
        // Consecutive threads read consecutive 4-float segments, so
        // global loads coalesce into full cache lines.
        // -----------------------------------------------------------------
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; ++i) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; ++j) {

                const int smem_row = thread_row + i;
                const int smem_col = thread_col + j;

                // A tile: rows from this block's row band,
                //         columns from this K phase.
                const int A_row = block_row + smem_row;
                const int A_col = phase * TILE_SIZE + smem_col;

                As[smem_row][smem_col] =
                    (A_row < M && A_col < K)
                        ? A[A_row * K + A_col]
                        : 0.0f;

                // B tile: rows from this K phase,
                //         columns from this block's column band.
                const int B_row = phase * TILE_SIZE + smem_row;
                const int B_col = block_col + smem_col;

                Bs[smem_row][smem_col] =
                    (B_row < K && B_col < N)
                        ? B[B_row * N + B_col]
                        : 0.0f;
            }
        }

        // Every thread must finish loading before any thread reads As/Bs.
        __syncthreads();

        // -----------------------------------------------------------------
        // Compute this phase
        //
        // For every k in the tile:
        //
        //   A contributes 4 values: As[thread_row + 0..3][k]
        //   B contributes 4 values: Bs[k][thread_col + 0..3]
        //
        // staged into registers, then 16 multiply-adds update the
        // 4x4 accumulator micro-tile.
        // -----------------------------------------------------------------
        for (int k = 0; k < TILE_SIZE; ++k) {

            float a[THREAD_TILE];
            float b[THREAD_TILE];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE; ++i) {
                a[i] = As[thread_row + i][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_TILE; ++j) {
                b[j] = Bs[k][thread_col + j];
            }

            // 4x4 register-blocked outer product accumulation.
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        // As/Bs are about to be overwritten in the next phase;
        // every thread must finish reading them first.
        __syncthreads();
    }

    // -----------------------------------------------------------------
    // Store the 4x4 register micro-tile into C
    // -----------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; ++j) {

            const int row = block_row + thread_row + i;
            const int col = block_col + thread_col + j;

            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

// =============================================================================
// Host launcher — the single owner of this kernel's launch configuration.
//
// Callers pass matrices and dimensions only; grid and block dims are derived
// here so a misconfigured launch is impossible.
// =============================================================================
void regblock_gemm(
    const float* A,
    const float* B,
    float*       C,
    int M,
    int N,
    int K
) {
    constexpr int TILE_SIZE       = 64;
    constexpr int THREADS_PER_DIM = 16;  // TILE_SIZE / THREAD_TILE

    dim3 block(THREADS_PER_DIM, THREADS_PER_DIM);          // 256 threads
    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    regblock_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
