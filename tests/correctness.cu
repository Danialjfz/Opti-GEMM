// =============================================================================
// Correctness tests — all CUDA kernels verified against CPU reference
//
// Philosophy:
//   The CPU implementation is the ground truth.
//   A kernel that produces wrong answers fast is not a fast kernel.
//   Every kernel must pass all tests here before it is benchmarked.
//
// Test cases:
//   - Square matrices (powers of two)
//   - Non-square matrices (M != N != K)
//   - Edge cases (1x1, single row, single column)
//   - Large matrices (stress correctness at scale)
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#include "cuda/cuda_check.cuh"
#include "cuda/gemm_naive.cuh"
// #include "cuda/gemm_tiled.cuh"      // uncomment as kernels are added
// #include "cuda/gemm_warptile.cuh"
// #include "cuda/gemm_regblock.cuh"

// =============================================================================
// CPU reference GEMM — naive triple loop, no optimization
// This is the correctness oracle. Never optimize this.
// =============================================================================

void cpu_gemm(
    const float* A,
    const float* B,
    float*       C,
    int M, int N, int K)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// =============================================================================
// Matrix utilities
// =============================================================================

// Fill matrix with predictable values for reproducible tests
void fill_matrix(float* mat, int rows, int cols, float seed = 1.0f)
{
    for (int i = 0; i < rows * cols; i++) {
        // Values in [-1, 1] — avoids overflow for large K
        mat[i] = seed * (2.0f * (static_cast<float>(i % 100) / 100.0f) - 1.0f);
    }
}

// Fill matrix with random values
void fill_random(float* mat, int size)
{
    for (int i = 0; i < size; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

// Check two matrices are equal within tolerance
// Returns true if they match, false otherwise
bool matrices_match(
    const float* ref,
    const float* got,
    int          M,
    int          N,
    float        abs_tol = 1e-4f,
    float        rel_tol = 1e-4f)
{
    int mismatches = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float abs_err = std::fabs(ref[i] - got[i]);
        float rel_err = abs_err / (std::fabs(ref[i]) + 1e-8f);

        if (abs_err > abs_tol && rel_err > rel_tol) {
            if (mismatches < 5) {
                // Print first few mismatches to help diagnose
                int row = i / N;
                int col = i % N;
                fprintf(stderr,
                    "  Mismatch at [%d][%d]: ref=%.6f  got=%.6f  "
                    "abs_err=%.2e  rel_err=%.2e\n",
                    row, col, ref[i], got[i], abs_err, rel_err);
            }
            mismatches++;
        }

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    if (mismatches == 0) {
        printf("    max_abs_err=%.2e  max_rel_err=%.2e\n",
               max_abs_err, max_rel_err);
    } else {
        fprintf(stderr, "    FAILED: %d mismatches (max_abs=%.2e  max_rel=%.2e)\n",
                mismatches, max_abs_err, max_rel_err);
    }

    return mismatches == 0;
}

// =============================================================================
// Generic kernel runner
// Takes a function pointer so the same test logic runs any kernel
// =============================================================================

using KernelFn = void (*)(const float*, const float*, float*, int, int, int);

bool run_kernel_test(
    KernelFn    kernel,
    int         M,
    int         N,
    int         K,
    const char* label)
{
    printf("  [%s]  M=%-5d N=%-5d K=%-5d  ... ", label, M, N, K);

    // -------------------------------------------------------------------------
    // Allocate host memory
    // -------------------------------------------------------------------------
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(M * N, 0.0f);   // CPU reference output
    std::vector<float> h_C_gpu(M * N, 0.0f);   // GPU kernel output

    fill_random(h_A.data(), M * K);
    fill_random(h_B.data(), K * N);

    // -------------------------------------------------------------------------
    // CPU reference
    // -------------------------------------------------------------------------
    cpu_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    // -------------------------------------------------------------------------
    // Allocate device memory
    // -------------------------------------------------------------------------
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
    // Launch kernel
    // 32x32 thread block — standard starting point for GEMM kernels
    // -------------------------------------------------------------------------
    constexpr int BLOCK_DIM = 32;
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(
        (N + BLOCK_DIM - 1) / BLOCK_DIM,
        (M + BLOCK_DIM - 1) / BLOCK_DIM
    );

    kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -------------------------------------------------------------------------
    // Copy result back and compare
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    bool passed = matrices_match(h_C_ref.data(), h_C_gpu.data(), M, N);

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed;
}

// =============================================================================
// Test suite definition
// =============================================================================

struct TestCase {
    int M, N, K;
    const char* description;
};

static const TestCase TEST_CASES[] = {
    // Edge cases
    {   1,   1,   1,  "1x1 trivial"          },
    {   1,  64,   1,  "single row"            },
    {  64,   1,   1,  "single column"         },
    {   1,   1,  64,  "K-only reduction"      },

    // Square — powers of two
    {  32,  32,  32,  "32x32 square"          },
    { 128, 128, 128,  "128x128 square"        },
    { 256, 256, 256,  "256x256 square"        },
    { 512, 512, 512,  "512x512 square"        },

    // Non-square — real workload shapes
    { 128, 256,  64,  "non-square M<N"        },
    { 256, 128,  64,  "non-square M>N"        },
    {  64, 128, 256,  "tall K"                },
    { 512, 256, 128,  "wide output"           },

    // Non-power-of-two — tests boundary handling
    { 127, 127, 127,  "127x127 odd"           },
    { 100, 200, 150,  "arbitrary shape"       },
    { 257, 511, 333,  "prime-ish dimensions"  },
};

// =============================================================================
// Run all tests for one kernel
// Returns number of failures
// =============================================================================

int test_kernel(KernelFn kernel, const char* kernel_name)
{
    printf("\n=== %s ===\n", kernel_name);

    int failures = 0;
    int total    = sizeof(TEST_CASES) / sizeof(TEST_CASES[0]);

    for (int i = 0; i < total; i++) {
        const TestCase& tc = TEST_CASES[i];
        bool passed = run_kernel_test(kernel, tc.M, tc.N, tc.K, tc.description);
        if (!passed) failures++;
    }

    printf("\n  Result: %d / %d passed\n", total - failures, total);
    return failures;
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    printf("MatrixEngine — Correctness Tests\n");
    printf("=================================\n");

    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d\n\n", prop.major, prop.minor);

    int total_failures = 0;

    // -------------------------------------------------------------------------
    // Test each kernel here as they are implemented
    // -------------------------------------------------------------------------

    total_failures += test_kernel(naive_gemm, "Naive GEMM");

    // Uncomment as kernels are added:
    // total_failures += test_kernel(tiled_gemm,    "Shared Memory Tiled GEMM");
    // total_failures += test_kernel(warptile_gemm, "Warp Tiled GEMM");
    // total_failures += test_kernel(regblock_gemm, "Register Blocked GEMM");

    // -------------------------------------------------------------------------
    // Final report
    // -------------------------------------------------------------------------
    printf("\n=================================\n");
    if (total_failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("FAILED: %d test(s) did not pass\n", total_failures);
    }
    printf("=================================\n");

    return total_failures > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}