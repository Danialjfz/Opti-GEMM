#pragma once

#include <cstdio>
#include <cstdlib>

// =============================================================================
// CUDA_CHECK — wraps any CUDA API call and aborts on failure
//
// Usage:
//   CUDA_CHECK(cudaMalloc(&ptr, size));
//   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
//
// Without this, failed CUDA calls return error codes silently.
// Silent failures produce wrong answers or crashes with no useful message.
// =============================================================================

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr,                                                 \
                    "CUDA error at %s:%d  →  %s\n",                        \
                    __FILE__, __LINE__,                                     \
                    cudaGetErrorString(err));                               \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)