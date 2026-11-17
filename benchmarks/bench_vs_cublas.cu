#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "cuda/cuda_check.cuh"
#include "cuda/gemm_naive.cuh"
#include "cuda/gemm_regblock.cuh"
#include "cuda/gemm_tiled.cuh"
#include "cuda/gemm_warptile.cuh"

namespace {

constexpr int WARMUP_ITERS = 10;
constexpr int REPEATS = 100;
constexpr int SAMPLES = 7;
constexpr float ATOL = 1e-4f;
constexpr float RTOL = 1e-4f;

const std::vector<int> TEST_SIZES = {
    128, 256, 512, 1024, 2048, 4096
};

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t status = (call);                                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                               \
            std::cerr << "cuBLAS error at " << __FILE__ << ":"              \
                      << __LINE__ << " (status " << status << ")\n";         \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (false)

using GemmLauncher = void (*)(
    const float*, const float*, float*, int, int, int);

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

void launch_naive(
    const float* A, const float* B, float* C, int M, int N, int K)
{
    dim3 block(32, 8);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    naive_gemm<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_tiled(
    const float* A, const float* B, float* C, int M, int N, int K)
{
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);
    tiled_gemm<<<grid, block>>>(A, B, C, M, N, K);
}

void cpu_gemm(
    const float* A, const float* B, float* C, int M, int N, int K)
{
    std::vector<double> accumulator(static_cast<size_t>(N));

    for (int row = 0; row < M; ++row) {
        std::fill(accumulator.begin(), accumulator.end(), 0.0);

        for (int k = 0; k < K; ++k) {
            const double a = A[static_cast<size_t>(row) * K + k];
            const float* b_row = B + static_cast<size_t>(k) * N;

            for (int col = 0; col < N; ++col) {
                accumulator[col] += a * b_row[col];
            }
        }

        for (int col = 0; col < N; ++col) {
            C[static_cast<size_t>(row) * N + col] =
                static_cast<float>(accumulator[col]);
        }
    }
}

bool matches_reference(
    const float* reference, const float* result, size_t count,
    const char* name)
{
    size_t errors = 0;
    float max_abs_error = 0.0f;

    for (size_t i = 0; i < count; ++i) {
        const float abs_error = std::fabs(reference[i] - result[i]);
        const float tolerance = ATOL + RTOL * std::fabs(reference[i]);
        max_abs_error = std::max(max_abs_error, abs_error);

        if (!std::isfinite(result[i]) || abs_error > tolerance) {
            if (errors < 5) {
                std::cerr << "Mismatch in " << name << " at index " << i
                          << ": expected=" << reference[i]
                          << ", got=" << result[i]
                          << ", abs_error=" << abs_error << '\n';
            }
            ++errors;
        }
    }

    if (errors != 0) {
        std::cerr << "[FAIL] " << name << ": " << errors << "/" << count
                  << " values differ, max_abs_error=" << max_abs_error << '\n';
        return false;
    }

    std::cout << "[PASS] " << name
              << " (max_abs_error=" << std::scientific << max_abs_error
              << std::defaultfloat << ")\n";
    return true;
}

template <typename Launch>
BenchResult time_launch(Launch&& launch, int M, int N, int K)
{
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> times;
    times.reserve(SAMPLES);

    for (int sample = 0; sample < SAMPLES; ++sample) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < REPEATS; ++i) {
            launch();
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        times.push_back(static_cast<double>(elapsed_ms) / REPEATS);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double mean =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double variance = 0.0;
    for (double time : times) {
        variance += (time - mean) * (time - mean);
    }
    variance /= times.size();

    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    const double gflops =
        (2.0 * M * N * K) / (median * 1.0e6);

    return {median, mean, std::sqrt(variance), gflops};
}

BenchResult time_cublas(
    cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major matrices. For row-major C=A*B, compute
    // column-major C^T = B^T*A^T by swapping A and B and using N-by-M C.
    auto launch = [&]() {
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N));
    };

    return time_launch(launch, M, N, K);
}

} // namespace

int main()
{
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "Opti-GEMM: handwritten kernels vs cuBLAS\n"
              << "GPU: " << prop.name << " (SM "
              << prop.major << "." << prop.minor << ")\n"
              << "Warmup: " << WARMUP_ITERS
              << ", repeats/sample: " << REPEATS
              << ", samples: " << SAMPLES << "\n\n";

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    const std::vector<KernelRef> kernels = {
        {"Naive", launch_naive},
        {"Tiled-SMEM", launch_tiled},
        {"Reg-Block", regblock_gemm},
        {"Warp-Tile", warptile_gemm},
    };

    for (int size : TEST_SIZES) {
        const int M = size;
        const int N = size;
        const int K = size;
        const size_t elements_a = static_cast<size_t>(M) * K;
        const size_t elements_b = static_cast<size_t>(K) * N;
        const size_t elements_c = static_cast<size_t>(M) * N;
        const size_t bytes_a = elements_a * sizeof(float);
        const size_t bytes_b = elements_b * sizeof(float);
        const size_t bytes_c = elements_c * sizeof(float);

        std::vector<float> h_A(elements_a);
        std::vector<float> h_B(elements_b);
        std::vector<float> h_reference(elements_c);
        std::vector<float> h_result(elements_c);

        for (float& value : h_A) {
            value = distribution(rng);
        }
        for (float& value : h_B) {
            value = distribution(rng);
        }
        cpu_gemm(h_A.data(), h_B.data(), h_reference.data(), M, N, K);

        float* d_A = nullptr;
        float* d_B = nullptr;
        float* d_C = nullptr;
        CUDA_CHECK(cudaMalloc(&d_A, bytes_a));
        CUDA_CHECK(cudaMalloc(&d_B, bytes_b));
        CUDA_CHECK(cudaMalloc(&d_C, bytes_c));
        CUDA_CHECK(cudaMemcpy(
            d_A, h_A.data(), bytes_a, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_B, h_B.data(), bytes_b, cudaMemcpyHostToDevice));

        std::cout << "\n=== " << size << " x " << size << " x " << size
                  << " ===\n"
                  << std::left << std::setw(16) << "Implementation"
                  << std::right << std::setw(14) << "Median (ms)"
                  << std::setw(14) << "Mean (ms)"
                  << std::setw(14) << "StdDev (ms)"
                  << std::setw(14) << "GFLOP/s"
                  << std::setw(12) << "vs cuBLAS" << '\n';

        std::vector<BenchResult> results;
        results.reserve(kernels.size() + 1);

        for (const KernelRef& kernel : kernels) {
            CUDA_CHECK(cudaMemset(d_C, 0, bytes_c));
            kernel.launch(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(
                h_result.data(), d_C, bytes_c, cudaMemcpyDeviceToHost));

            if (!matches_reference(
                    h_reference.data(), h_result.data(), elements_c,
                    kernel.name)) {
                CUDA_CHECK(cudaFree(d_A));
                CUDA_CHECK(cudaFree(d_B));
                CUDA_CHECK(cudaFree(d_C));
                CUBLAS_CHECK(cublasDestroy(handle));
                return EXIT_FAILURE;
            }

            results.push_back(time_launch(
                [&]() {
                    kernel.launch(d_A, d_B, d_C, M, N, K);
                },
                M, N, K));
        }

        CUDA_CHECK(cudaMemset(d_C, 0, bytes_c));
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            h_result.data(), d_C, bytes_c, cudaMemcpyDeviceToHost));

        if (!matches_reference(
                h_reference.data(), h_result.data(), elements_c, "cuBLAS")) {
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
            CUBLAS_CHECK(cublasDestroy(handle));
            return EXIT_FAILURE;
        }

        results.push_back(time_cublas(
            handle, d_A, d_B, d_C, M, N, K));

        const double cublas_gflops = results.back().gflops;
        for (size_t i = 0; i < results.size(); ++i) {
            const char* name = i < kernels.size()
                ? kernels[i].name
                : "cuBLAS";
            std::cout << std::left << std::setw(16) << name
                      << std::right << std::fixed << std::setprecision(4)
                      << std::setw(14) << results[i].median_ms
                      << std::setw(14) << results[i].mean_ms
                      << std::setw(14) << results[i].stddev_ms
                      << std::setw(14) << std::setprecision(2)
                      << results[i].gflops
                      << std::setw(12) << std::setprecision(2)
                      << results[i].gflops / cublas_gflops << "x\n";
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return EXIT_SUCCESS;
}
