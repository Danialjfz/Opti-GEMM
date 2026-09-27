
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ============================================================
// Opti-GEMM: Kernel Benchmark
// Target: NVIDIA Tesla T4 (SM 7.5)
// Precision: FP32
// Operation: C[M,N] = A[M,K] * B[K,N]
// Layout: Row-major
// ============================================================

// ------------------------------------------------------------
// 1. Include your kernel headers
// ------------------------------------------------------------

// Adjust these paths/names to match your repository.
#include "cuda/gemm_naive.cuh"
#include "cuda/gemm_tiled.cuh"
#include "cuda/gemm_regblock.cuh"

// Add the warp-tiled header once implemented.
// #include "cuda/gemm_warptile.cuh"


// ------------------------------------------------------------
// 2. CUDA error checking
// ------------------------------------------------------------

#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = (call);                                    \
        if (err != cudaSuccess) {                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":"         \
                      << __LINE__ << " -> "                           \
                      << cudaGetErrorString(err) << std::endl;       \
            std::exit(EXIT_FAILURE);                                 \
        }                                                            \
    } while (0)


// ------------------------------------------------------------
// 3. Benchmark configuration
// ------------------------------------------------------------

constexpr int WARMUP_ITERS = 10;
constexpr int REPEATS       = 100;
constexpr int SAMPLES       = 7;

constexpr float ATOL = 1e-4f;
constexpr float RTOL = 1e-4f;

// Matrix sizes to benchmark.
// These are square GEMMs: M=N=K.
const std::vector<int> TEST_SIZES = {
    128,
    256,
    512,
    1024,
    2048,
    4096
};


// ------------------------------------------------------------
// 4. Common kernel launcher interface
// ------------------------------------------------------------

// Every implementation must provide a host-side function
// with this signature.
//
// The host function launches its own CUDA kernel using its
// own grid and block dimensions.
//
// IMPORTANT:
// Do NOT use a CUDA global kernel function pointer here.
// CUDA's <<<grid, block>>> launch is performed inside the
// corresponding host-side launcher.

using GemmLauncher = void (*)(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
);

struct KernelRef {
    const char* name;
    GemmLauncher launch;
};

struct BenchResult {
    double median_ms;
    double mean_ms;
    double stddev_ms;
    double gflops;
};


// ------------------------------------------------------------
// 5. Kernel launchers
// ------------------------------------------------------------

// These launchers assume that your kernel headers expose
// the following CUDA global kernels:
//
// naive_gemm(A, B, C, M, N, K)
// tiled_gemm(A, B, C, M, N, K)
//
// If your actual global kernel names differ, change only
// the names in these launcher functions.

// ---------------- Naive GEMM ----------------

void launch_naive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    // 256 threads per block.
    // Each warp covers 32 adjacent output columns.

    dim3 block(32, 8);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    naive_gemm<<<grid, block>>>(
        A, B, C, M, N, K
    );
}


// ---------------- Shared-memory GEMM ----------------

void launch_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    // This configuration assumes TILE_SIZE=16
    // and a 16x16 thread block in your tiled kernel.

    constexpr int TILE = 16;

    dim3 block(TILE, TILE);

    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    tiled_gemm<<<grid, block>>>(
        A, B, C, M, N, K
    );
}


// ---------------- Register-blocked GEMM ----------------

// IMPORTANT:
// This launcher must match YOUR implemented register-blocked
// kernel's thread-to-output mapping.
//
// The configuration below is a placeholder.
// Replace the block and grid calculations and kernel name
// with the actual implementation in gemm_regblock.cu.
//
// If your header already provides a host wrapper called
// regblock_gemm(A,B,C,M,N,K), call that wrapper here instead.

void launch_regblock(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    constexpr int TILE_SIZE = 16;

    dim3 block(4, 4);

    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    regblock_gemm<<<grid, block>>>(
        A, B, C, M, N, K
    );
}

// ------------------------------------------------------------
// 6. Register kernels in the benchmark
// ------------------------------------------------------------

std::vector<KernelRef> get_kernels()
{
    return {
        {"Naive",      launch_naive},
        {"Tiled-SMEM", launch_tiled},
        {"Reg-Block",  launch_regblock},

        // Add when implemented:
        // {"Warp-Tile", launch_warptile},
    };
}


// ------------------------------------------------------------
// 7. CPU reference GEMM
// ------------------------------------------------------------

// Row-major reference implementation.
// Computes C[M,N] = A[M,K] * B[K,N].

void cpu_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}


// ------------------------------------------------------------
// 8. Correctness validation
// ------------------------------------------------------------

bool validate_result(
    const float* reference,
    const float* result,
    int M,
    int N,
    const char* kernel_name)
{
    const size_t total =
        static_cast<size_t>(M) * N;

    double max_abs_error = 0.0;
    double max_rel_error = 0.0;

    size_t errors = 0;

    for (size_t i = 0; i < total; ++i) {

        const float ref = reference[i];
        const float got = result[i];

        const float abs_error = std::fabs(ref - got);

        const float tolerance =
            ATOL + RTOL * std::fabs(ref);

        if (!std::isfinite(got) ||
            !std::isfinite(ref) ||
            abs_error > tolerance)
        {
            if (errors < 5) {
                std::cerr
                    << "Mismatch at index " << i
                    << ": expected=" << ref
                    << ", got=" << got
                    << ", abs_error=" << abs_error
                    << std::endl;
            }

            ++errors;
        }

        max_abs_error = std::max(
            max_abs_error,
            static_cast<double>(abs_error)
        );

        const double denom =
            std::max(std::fabs(static_cast<double>(ref)),
                     1e-8);

        max_rel_error = std::max(
            max_rel_error,
            abs_error / denom
        );
    }

    if (errors > 0) {
        std::cerr
            << "[FAIL] " << kernel_name
            << " | Errors: " << errors
            << "/" << total
            << " | Max abs error: " << max_abs_error
            << " | Max rel error: " << max_rel_error
            << std::endl;

        return false;
    }

    std::cout
        << "[PASS] " << kernel_name
        << " | Max abs error: "
        << std::scientific << max_abs_error
        << std::defaultfloat
        << std::endl;

    return true;
}


// ------------------------------------------------------------
// 9. Benchmark timing
// ------------------------------------------------------------

BenchResult run_benchmark(
    const KernelRef& kernel,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K)
{
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup launches.

    for (int i = 0; i < WARMUP_ITERS; ++i) {

        kernel.launch(
            d_A, d_B, d_C, M, N, K
        );

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> times;
    times.reserve(SAMPLES);

    // Repeated batches to reduce event timing overhead.

    for (int sample = 0; sample < SAMPLES; ++sample) {

        CUDA_CHECK(cudaEventRecord(start));

        for (int i = 0; i < REPEATS; ++i) {

            kernel.launch(
                d_A, d_B, d_C, M, N, K
            );

            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop
        ));

        const double avg_ms =
            static_cast<double>(elapsed_ms) / REPEATS;

        times.push_back(avg_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Mean.

    const double sum =
        std::accumulate(
            times.begin(),
            times.end(),
            0.0
        );

    const double mean =
        sum / times.size();

    // Population standard deviation.

    double variance = 0.0;

    for (double t : times) {
        variance += (t - mean) * (t - mean);
    }

    variance /= times.size();

    const double stddev =
        std::sqrt(variance);

    // Median.

    std::sort(times.begin(), times.end());

    const double median =
        times[times.size() / 2];

    // Conventional FP32 GEMM FLOP count:
    // 2 * M * N * K.

    const double flops =
        2.0 * M * N * K;

    const double gflops =
        flops / (median * 1e6);

    return {
        median,
        mean,
        stddev,
        gflops
    };
}


// ------------------------------------------------------------
// 10. Main
// ------------------------------------------------------------

int main()
{
    std::cout
        << "========================================\n"
        << "Opti-GEMM Benchmark\n"
        << "========================================\n";

    // Query the active CUDA device.

    int device_id = 0;

    CUDA_CHECK(cudaGetDevice(&device_id));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(
        &prop,
        device_id
    ));

    std::cout
        << "GPU: " << prop.name << "\n"
        << "Compute capability: "
        << prop.major << "."
        << prop.minor << "\n"
        << "SM count: "
        << prop.multiProcessorCount << "\n"
        << "Warmup iterations: "
        << WARMUP_ITERS << "\n"
        << "Repeats per sample: "
        << REPEATS << "\n"
        << "Samples: "
        << SAMPLES << "\n\n";

    // This benchmark uses the largest configured dimension
    // to allocate reusable device buffers.

    const int MAX_DIM =
        *std::max_element(
            TEST_SIZES.begin(),
            TEST_SIZES.end()
        );

    const size_t MAX_ELEMENTS =
        static_cast<size_t>(MAX_DIM) * MAX_DIM;

    const size_t BYTES =
        MAX_ELEMENTS * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_A, BYTES
    ));

    CUDA_CHECK(cudaMalloc(
        &d_B, BYTES
    ));

    CUDA_CHECK(cudaMalloc(
        &d_C, BYTES
    ));

    // Reproducible host inputs.

    std::vector<float> h_A(MAX_ELEMENTS);
    std::vector<float> h_B(MAX_ELEMENTS);

    std::mt19937 rng(12345);

    std::uniform_real_distribution<float> dist(
        -1.0f, 1.0f
    );

    for (float& x : h_A) {
        x = dist(rng);
    }

    for (float& x : h_B) {
        x = dist(rng);
    }

    // Copy once, outside the timed region.

    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A.data(),
        BYTES,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B.data(),
        BYTES,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemset(
        d_C, 0, BYTES
    ));

    const auto kernels =
        get_kernels();

    bool all_passed = true;

    // --------------------------------------------------------
    // Benchmark every configured matrix size.
    // --------------------------------------------------------

    for (int N : TEST_SIZES) {

        const int M = N;
        const int K = N;

        const size_t elements =
            static_cast<size_t>(M) * N;

        const size_t bytes =
            elements * sizeof(float);

        std::cout
            << "\n========================================\n"
            << "Problem: M=" << M
            << ", N=" << N
            << ", K=" << K
            << "\n========================================\n";

        // For square matrices, the first N*N elements of
        // each maximum-sized buffer are contiguous and can
        // be interpreted as an N x N row-major matrix.

        std::vector<float> h_ref(elements);
        std::vector<float> h_out(elements);

        // CPU reference.

        std::cout
            << "Computing CPU reference..."
            << std::endl;

        cpu_gemm(
            h_A.data(),
            h_B.data(),
            h_ref.data(),
            M, N, K
        );

        std::cout
            << "CPU reference complete."
            << std::endl;

        // Copy the relevant contiguous input regions
        // into the device buffers.

        CUDA_CHECK(cudaMemcpy(
            d_A,
            h_A.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_B,
            h_B.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        std::cout
            << std::left
            << std::setw(16) << "Kernel"
            << std::right
            << std::setw(14) << "Median(ms)"
            << std::setw(14) << "Mean(ms)"
            << std::setw(14) << "StdDev(ms)"
            << std::setw(16) << "GFLOP/s"
            << "\n";

        std::cout
            << std::string(74, '-')
            << "\n";

        // ----------------------------------------------------
        // Each kernel: correctness first, then performance.
        // ----------------------------------------------------

        for (const auto& kernel : kernels) {

            std::cout
                << "\nTesting "
                << kernel.name
                << "..."
                << std::endl;

            CUDA_CHECK(cudaMemset(
                d_C, 0, bytes
            ));

            // One launch before validation.

            kernel.launch(
                d_A, d_B, d_C,
                M, N, K
            );

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(
                h_out.data(),
                d_C,
                bytes,
                cudaMemcpyDeviceToHost
            ));

            const bool passed =
                validate_result(
                    h_ref.data(),
                    h_out.data(),
                    M,
                    N,
                    kernel.name
                );

            if (!passed) {
                all_passed = false;

                std::cerr
                    << "Skipping benchmark for "
                    << kernel.name
                    << " because correctness failed."
                    << std::endl;

                continue;
            }

            // Only benchmark kernels that passed validation.

            std::cout
                << "Benchmarking "
                << kernel.name
                << "..."
                << std::endl;

            const BenchResult result =
                run_benchmark(
                    kernel,
                    d_A, d_B, d_C,
                    M, N, K
                );

            std::cout
                << std::left
                << std::setw(16)
                << kernel.name
                << std::right
                << std::fixed
                << std::setprecision(4)
                << std::setw(14)
                << result.median_ms
                << std::setw(14)
                << result.mean_ms
                << std::setw(14)
                << result.stddev_ms
                << std::setw(16)
                << std::setprecision(2)
                << result.gflops
                << "\n";
        }
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::cout
        << "\n========================================\n";

    if (all_passed) {
        std::cout
            << "All tested kernels passed correctness.\n";
    } else {
        std::cout
            << "WARNING: One or more kernels failed "
            << "correctness or are not implemented.\n";
    }

    std::cout
        << "Benchmark complete.\n"
        << "========================================\n";

    return all_passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}