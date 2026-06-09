// =============================================================================
// Benchmark — Naive GEMM
//
// Timing method:  CUDA events (measures GPU execution time only)
// Warmup:         5 iterations before any timing
// Timed runs:     20 iterations, report mean / min / max / stddev
// Metric:         GFLOP/s and % of cuBLAS peak (added in Stage 5)
//
// Run this after correctness.cu passes all tests.
// Never benchmark a kernel that has not been verified correct.
// =============================================================================

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "cuda/cuda_check.cuh"
#include "cuda/gemm_naive.cuh"

// =============================================================================
// Timing utilities
// =============================================================================

// GFLOP count for an M x N x K GEMM
// Each output element requires K multiplications and K additions = 2K FLOP
inline double gemm_gflops(int M, int N, int K, double ms)
{
    double flop = 2.0 * M * N * K;
    double gflop = flop / 1e9;
    double seconds = ms / 1e3;
    return gflop / seconds;
}

struct BenchResult {
    double mean_ms;
    double min_ms;
    double max_ms;
    double stddev_ms;
    double gflops;
};

// =============================================================================
// Single benchmark run for one matrix size
// =============================================================================

BenchResult benchmark_naive_gemm(int M, int N, int K)
{
    constexpr int WARMUP_ITERS = 5;
    constexpr int TIMED_ITERS  = 20;
    constexpr int BLOCK_DIM    = 32;

    // -------------------------------------------------------------------------
    // Allocate and initialize
    // -------------------------------------------------------------------------
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    // -------------------------------------------------------------------------
    // Launch config
    // -------------------------------------------------------------------------
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(
        (N + BLOCK_DIM - 1) / BLOCK_DIM,
        (M + BLOCK_DIM - 1) / BLOCK_DIM
    );

    // -------------------------------------------------------------------------
    // CUDA events for timing
    // -------------------------------------------------------------------------
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // -------------------------------------------------------------------------
    // Warmup — bring GPU to steady state, populate caches
    // Do NOT time these
    // -------------------------------------------------------------------------
    for (int i = 0; i < WARMUP_ITERS; i++) {
        naive_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------------------
    // Timed iterations
    // -------------------------------------------------------------------------
    std::vector<double> times_ms(TIMED_ITERS);

    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(ev_start));

        naive_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        times_ms[i] = static_cast<double>(ms);
    }

    // -------------------------------------------------------------------------
    // Compute statistics
    // -------------------------------------------------------------------------
    double sum  = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    double mean = sum / TIMED_ITERS;
    double min  = *std::min_element(times_ms.begin(), times_ms.end());
    double max  = *std::max_element(times_ms.begin(), times_ms.end());

    double sq_sum = 0.0;
    for (double t : times_ms) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / TIMED_ITERS);

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return BenchResult {
        mean,
        min,
        max,
        stddev,
        gemm_gflops(M, N, K, mean)
    };
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    printf("MatrixEngine — Naive GEMM Benchmark\n");
    printf("=====================================\n\n");

    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device:  %s\n", prop.name);
    printf("Compute: %d.%d\n", prop.major, prop.minor);

    // Theoretical peak FLOP/s for reference (FP32, no tensor cores)
    // You will fill this in from your GPU's spec sheet
    // Example for RTX 3080: 29.77 TFLOP/s = 29770 GFLOP/s
    const double THEORETICAL_PEAK_GFLOPS = 0.0;   // TODO: set for your GPU

    printf("\n");
    printf("%-10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
           "Size", "Mean(ms)", "Min(ms)", "Max(ms)", "Std(ms)",
           "GFLOP/s", "% Peak");
    printf("%-10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
           "----------", "----------", "----------", "----------",
           "----------", "----------", "----------");

    // -------------------------------------------------------------------------
    // Benchmark sizes
    // -------------------------------------------------------------------------
    struct Size { int M, N, K; };
    static const Size SIZES[] = {
        {  256,  256,  256 },
        {  512,  512,  512 },
        { 1024, 1024, 1024 },
        { 2048, 2048, 2048 },
        { 4096, 4096, 4096 },
    };

    for (const auto& s : SIZES) {
        BenchResult r = benchmark_naive_gemm(s.M, s.N, s.K);

        double pct_peak = (THEORETICAL_PEAK_GFLOPS > 0.0)
                        ? (r.gflops / THEORETICAL_PEAK_GFLOPS * 100.0)
                        : 0.0;

        char size_label[32];
        snprintf(size_label, sizeof(size_label), "%dx%d", s.M, s.N);

        printf("%-10s  %10.3f  %10.3f  %10.3f  %10.3f  %10.2f  %9.2f%%\n",
               size_label,
               r.mean_ms,
               r.min_ms,
               r.max_ms,
               r.stddev_ms,
               r.gflops,
               pct_peak);
    }

    printf("\nNote: % Peak uses FP32 theoretical peak, no Tensor Cores.\n");
    printf("Set THEORETICAL_PEAK_GFLOPS for your GPU to see accurate numbers.\n");

    return EXIT_SUCCESS;
}