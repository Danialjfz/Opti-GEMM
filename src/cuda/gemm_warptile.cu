#include "cuda/gemm_warptile.cuh"

// =============================================================================
// Warp-tiled GEMM
//
//   Block tile   : 128 x 128 output elements
//   Thread block : 256 threads = 8 full warps, arranged 4 warps x 2 warps
//   Warp tile    : 32 x 64 output elements
//   Thread tile  : 8 x 8 output elements per thread (64 accumulators)
//   K-tile       : 8
//
// Why this configuration
// ----------------------
//
// The register-blocked kernel reached 64x64 outputs per block with a 4x4
// micro-tile. Each thread does only 16 FMAs per shared-memory load, and
// the block reuses each loaded element across 64 outputs. Warp tiling
// fixes both:
//
//   1. Block tile doubles to 128x128. Global loads are amortized over 4x
//      more compute, so arithmetic intensity per block jumps from 16
//      FLOP/byte to 32 FLOP/byte.
//
//   2. Explicit warp decomposition. Warps cooperate on 32x64 sub-tiles
//      before threads split to their own micro-tiles. This gives the
//      kernel a three-level hierarchy (block -> warp -> thread) instead
//      of the flat two-level structure of regblock. In practice this is
//      the structural prerequisite for the two optimizations that
//      follow: vectorized (float4) shared loads and cp.async double
//      buffering.
//
//   3. Thread micro-tile grows to 8x8 = 64 accumulators. Each thread now
//      performs 64 FMAs per shared-memory load (vs 16 in regblock).
//
// Why 4x2 warp arrangement (not 2x4 or 1x8)?
// ------------------------------------------
// Warp arrangement affects L2 locality. With warps stacked 4 rows x 2
// columns, adjacent warps in the same warp row share the same A rows
// and read different B columns -- good B reuse in L2. A 2x4 arrangement
// would break A reuse instead. 4x2 is the standard.
//
// Why BK=8 (not 16 or 32)?
// ------------------------
// BK controls the ratio of shared-memory traffic to compute per phase.
// Large BK means bigger shared tiles and less __syncthreads() overhead,
// but also more pressure on shared memory. BK=8 keeps As at 128x9 floats
// (with padding, ~4.6 KB) and Bs at 8x128 floats (~4.2 KB) -- ~8.7 KB
// shared per block. That leaves room for a second (double-buffered)
// copy of both tiles in a later stage.
//
// Register pressure trade-off
// ---------------------------
// 64 accumulators + 8 aFrag + 8 bFrag + addressing is roughly 90-110
// registers per thread. On a modern SM (~65,536 registers) that allows
// 2 resident blocks per SM (512 threads, ~25% occupancy). This is LOWER
// than regblock's occupancy and it is intentional: the 64 FMAs per
// shared load per thread give enough ILP to keep the FP units busy even
// at low thread count. Occupancy is not the goal; sustained FLOP/s is.
// =============================================================================

static constexpr int W_BM          = 128;               // block tile rows
static constexpr int W_BN          = 128;               // block tile cols
static constexpr int W_BK          = 8;                 // K-tile depth
static constexpr int W_TM          = 8;                 // thread micro-tile rows
static constexpr int W_TN          = 8;                 // thread micro-tile cols
static constexpr int W_NUM_THREADS = 256;               // threads per block
static constexpr int W_WARP_ROWS   = 4;                 // warps along row direction
static constexpr int W_WARP_COLS   = 2;                 // warps along col direction
static constexpr int W_WM          = W_BM / W_WARP_ROWS;   // 32 (warp tile rows)
static constexpr int W_WN          = W_BN / W_WARP_COLS;   // 64 (warp tile cols)
static constexpr int W_THREAD_ROWS = W_WM / W_TM;          // 4 thread rows per warp
static constexpr int W_THREAD_COLS = W_WN / W_TN;          // 8 thread cols per warp
static constexpr int W_AS_PAD      = 1;                 // 1 extra col on As, see below

static_assert(W_THREAD_ROWS * W_THREAD_COLS == 32,
              "each warp must contain exactly 32 threads");

__global__ void __launch_bounds__(W_NUM_THREADS)
warptile_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M,
    int N,
    int K
) {
    // Shared tiles.
    //
    // As is [BM][BK + pad] -- outer dim is A's row band, inner dim is the
    //   K phase. Padding is required here: without it (BK = 8), the four
    //   thread-row groups within a warp all hit the same bank when
    //   reading As[.][k]. A pad of 1 shifts the row stride to 9, which
    //   spreads those 4 reads across four distinct banks.
    //
    // Bs is [BK][BN]. Because BN = 128 is a multiple of 32, bank =
    //   column index mod 32. A warp reading 8 contiguous columns of
    //   Bs[k][.] hits 8 distinct banks with a 2-way broadcast per bank
    //   (4 thread rows read the same 8 addresses). No padding needed.
    __shared__ float As[W_BM][W_BK + W_AS_PAD];
    __shared__ float Bs[W_BK][W_BN];

    const int tid    = threadIdx.x;
    const int laneId = tid % 32;
    const int warpId = tid / 32;

    // Warp -> sub-tile of the 128x128 block output.
    const int warpRow = warpId / W_WARP_COLS;
    const int warpCol = warpId % W_WARP_COLS;

    // Lane -> micro-tile of the warp's 32x64 sub-tile.
    const int threadRow = laneId / W_THREAD_COLS;
    const int threadCol = laneId % W_THREAD_COLS;

    // Global start position of this block's C tile.
    const int blockRow = blockIdx.y * W_BM;
    const int blockCol = blockIdx.x * W_BN;

    // 64 registers per thread: 8x8 output micro-tile.
    float acc[W_TM][W_TN] = {0.0f};

    // Loop over K in BK-sized phases.
    const int numPhases = (K + W_BK - 1) / W_BK;
    for (int phase = 0; phase < numPhases; ++phase) {
        const int kPhase = phase * W_BK;

        // -----------------------------------------------------------------
        // Cooperative load of As and Bs.
        //
        // As[r][k] = A[blockRow + r][kPhase + k]
        //   Consecutive tids map to consecutive k within a row, so the
        //   global reads coalesce into full cache lines per A row.
        //
        // Bs[k][c] = B[kPhase + k][blockCol + c]
        //   Consecutive tids map to consecutive c within a k row, so
        //   the global reads coalesce cleanly into full cache lines per
        //   B row (128 floats = 512 bytes per row).
        // -----------------------------------------------------------------
        #pragma unroll
        for (int i = tid; i < W_BM * W_BK; i += W_NUM_THREADS) {
            const int r  = i / W_BK;
            const int k  = i % W_BK;
            const int gr = blockRow + r;
            const int gk = kPhase + k;
            As[r][k] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
        }
        #pragma unroll
        for (int i = tid; i < W_BK * W_BN; i += W_NUM_THREADS) {
            const int k  = i / W_BN;
            const int c  = i % W_BN;
            const int gk = kPhase + k;
            const int gc = blockCol + c;
            Bs[k][c] = (gk < K && gc < N) ? B[gk * N + gc] : 0.0f;
        }

        __syncthreads();

        // -----------------------------------------------------------------
        // Compute this phase.
        //
        // For every k in the tile:
        //   aFrag[i] = As[warpRow*WM + threadRow*TM + i][k]   (i in 0..7)
        //   bFrag[j] = Bs[k][warpCol*WN + threadCol*TN + j]   (j in 0..7)
        // then 64 outer-product FMAs update this thread's 8x8 micro-tile.
        //
        // The k loop is fully unrolled (BK = 8) so that aFrag/bFrag
        // register allocation can be tightened by the compiler and the
        // 64-FMA blocks chain without any loop-carried dependencies.
        // -----------------------------------------------------------------
        #pragma unroll
        for (int k = 0; k < W_BK; ++k) {
            float aFrag[W_TM];
            float bFrag[W_TN];

            #pragma unroll
            for (int i = 0; i < W_TM; ++i) {
                aFrag[i] = As[warpRow * W_WM + threadRow * W_TM + i][k];
            }
            #pragma unroll
            for (int j = 0; j < W_TN; ++j) {
                bFrag[j] = Bs[k][warpCol * W_WN + threadCol * W_TN + j];
            }

            #pragma unroll
            for (int i = 0; i < W_TM; ++i) {
                #pragma unroll
                for (int j = 0; j < W_TN; ++j) {
                    acc[i][j] += aFrag[i] * bFrag[j];
                }
            }
        }

        // As/Bs are about to be overwritten in the next phase; every
        // thread must finish reading them first.
        __syncthreads();
    }

    // -----------------------------------------------------------------
    // Store the 8x8 register micro-tile into C.
    // -----------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < W_TM; ++i) {
        #pragma unroll
        for (int j = 0; j < W_TN; ++j) {
            const int gr = blockRow + warpRow * W_WM + threadRow * W_TM + i;
            const int gc = blockCol + warpCol * W_WN + threadCol * W_TN + j;
            if (gr < M && gc < N) {
                C[gr * N + gc] = acc[i][j];
            }
        }
    }
}

// =============================================================================
// Host launcher -- the single owner of this kernel's launch configuration.
//
// Callers pass matrices and dimensions only; grid and block dims are
// derived here so a misconfigured launch is impossible.
// =============================================================================
void warptile_gemm(
    const float* A,
    const float* B,
    float*       C,
    int M,
    int N,
    int K
) {
    dim3 block(W_NUM_THREADS);
    dim3 grid(
        (N + W_BN - 1) / W_BN,
        (M + W_BM - 1) / W_BM
    );
    warptile_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}