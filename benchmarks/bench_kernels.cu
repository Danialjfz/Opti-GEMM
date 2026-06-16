/** 
 * Opti-GEMM Advanced Benchmarking Suite
 * 
 * DESIGN PRINCIPLES:
 * 1. ZERO-ALLOC LOOP: Memory is allocated once at max size to avoid driver latency.
 * 2. ARCHITECTURE AWARE: Peak GFLOPs are calculated based on the specific GPU detected.
 * 3. COLD/HOT SEPARATION: Warmup iterations ensure we measure hardware in a steady state.
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>

// Header for error handling (macros like CUDA_CHECK)
#include "cuda/cuda_check.cuh"

// Import all kernel variations for comparison
#include "cuda/gemm_naive.cuh"
#include "cuda/gemm_tiled.cuh"
#include "cuda/gemm_warptile.cuh"
#include "cuda/gemm_regblock.cuh"

// =============================================================================
// HARDWARE QUERIES
// =============================================================================

/**
 * Returns the number of single-precision (FP32) CUDA cores per Streaming Multiprocessor (SM).
 * This varies by architecture (Pascal, Ampere, Ada, etc.).
 */
inline int get_cores_per_sm(int major, int minor) {
    if (major == 7) return 64;                          // Volta & Turing
    if (major == 8) return (minor == 0) ? 64 : 128;     // Ampere (A100=64, RTX 30-series=128)
    if (major >= 9) return 128;                         // Hopper & Ada Lovelace
    return 128;                                         // Default for future/unknown
}

/**
 * Calculates theoretical max throughput in GFLOP/s.
 * Formula: SMs * CoresPerSM * ClockRate(GHz) * 2 (for FMA - Fused Multiply Add)
 */
inline double get_theoretical_peak(const cudaDeviceProp& prop) {
    double clock_ghz = prop.clockRate / 1e6; // clockRate is in kHz
    int cores_per_sm = get_cores_per_sm(prop.major, prop.minor);
    return prop.multiProcessorCount * cores_per_sm * clock_ghz * 2.0;
}

// =============================================================================
// BENCHMARKING CORE
// =============================================================================

// Define a function pointer type that matches all GEMM kernel signatures
typedef void (*gemm_kernel_t)(const float*, const float*, float*, int, int, int);

// Helper struct to hold kernel metadata
struct KernelRef {
    std::string name;
    gemm_kernel_t fn;
    int bx;
    int by;
};

struct BenchResult {
    std::string name;
    double mean_ms, stddev_ms, gflops;
};

/**
 * Executes a specific kernel multiple times and records performance metrics.
 */
BenchResult run_benchmark(
    const std::string& name,
    gemm_kernel_t kernel,
    float *d_A, float *d_B, float *d_C,
    int M, int N, int K,
    int block_x, int block_y
) {
    const int WARMUP_ITERS = 5;  // Ignore the first few runs (JIT overhead, clock ramp-up)
    const int TIMED_ITERS  = 20; // Number of samples for statistical significance

    dim3 block(block_x, block_y);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARMUP PHASE ---
    // Why? First kernel launches involve driver overhead and hardware "waking up".
    // We also want to pre-load the L2 cache if the problem size is small.
    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    // Block CPU until GPU is finished with all warmup tasks
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- MEASUREMENT PHASE ---
    std::vector<float> times(TIMED_ITERS);
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        
        CUDA_CHECK(cudaEventRecord(stop));
        
        // Synchronize stop event specifically to avoid CPU overhead of deviceSync
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[i] = ms;
    }

    // Statistics Calculation
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / TIMED_ITERS;
    
    double variance = 0.0;
    for(float t : times) variance += (t - mean) * (t - mean);
    double stddev = std::sqrt(variance / TIMED_ITERS);

    // GFLOP/s = (2 * M * N * K) / (time_in_seconds * 10^9)
    double gflops = (2.0 * M * N * K) / (mean * 1e6);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return {name, mean, stddev, gflops};
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

int main() {
    // 1. Hardware Discovery
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak = get_theoretical_peak(prop);

    printf(">>> RUNNING GEMM BENCHMARK SUITE <<<\n");
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Theoretical Peak: %.2f GFLOP/s (FP32)\n\n", peak);

    // 2. Pre-allocate Global Memory
    // We allocate once for the largest possible problem size. 
    // Re-allocating inside the loop causes significant "noise" in benchmarks.
    const int MAX_DIM = 4096;
    size_t bytes = MAX_DIM * MAX_DIM * sizeof(float);
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 3. Define the Kernels to Test
    // Add your new optimized kernels to this list as you develop them.
    std::vector<KernelRef> kernels = {
        {"Naive",           naive_gemm,       32, 32},
        {"Tiled-SMEM",      tiled_gemm,       16, 16},
        {"Warp-Tile",       warptile_gemm,    32, 8},   // adjust block dims to your kernel
        {"Reg-Block",       regblock_gemm,    64, 4}    // check your implementation
    };

    // 4. Define Problem Sizes
    // Powers of 2 are standard; they reveal cache alignment behavior.
    std::vector<int> test_sizes = {512, 1024, 2048, 4096};

    // 5. Execution Loop
    for (int N : test_sizes) {
        printf("--- PROBLEM SIZE: %d x %d x %d ---\n", N, N, N);
        printf("%-15s | %-10s | %-10s | %-12s | %-10s\n", 
               "Algorithm", "Time(ms)", "StdDev", "GFLOP/s", "Efficiency");
        printf("----------------------------------------------------------------------\n");

        for (auto& k : kernels) {
            auto r = run_benchmark(k.name, k.fn, d_A, d_B, d_C, N, N, N, k.bx, k.by);
            
            double efficiency = (r.gflops / peak) * 100.0;
            printf("%-15s | %10.3f | %10.4f | %12.2f | %9.2f%%\n", 
                   r.name.c_str(), r.mean_ms, r.stddev_ms, r.gflops, efficiency);
        }
        printf("\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}