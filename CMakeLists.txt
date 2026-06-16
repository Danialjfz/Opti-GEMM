# =============================================================================
# Opti-Gemm CMakeLists.txt
# =============================================================================

cmake_minimum_required(VERSION 3.18)

project(Opti-GEMM VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# Options
# =============================================================================

option(OPTI-GEMM_ENABLE_CUDA "Enable CUDA targets" OFF)

# =============================================================================
# Standards
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# CPU Matrix Library
# =============================================================================

add_library(matrixlib
    ${CMAKE_CURRENT_SOURCE_DIR}/src/MatrixLib.cpp
)

target_include_directories(matrixlib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# =============================================================================
# CPU Tests
# =============================================================================

add_executable(test_matrix
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_matrix.cpp
)

target_link_libraries(test_matrix PRIVATE
    matrixlib
)

# =============================================================================
# CPU Benchmark
# =============================================================================

add_executable(benchmark_gemm
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/benchmark_gemm.cpp
)

target_link_libraries(benchmark_gemm PRIVATE
    matrixlib
)

# =============================================================================
# Examples
# =============================================================================

add_executable(example_basic
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/basic_operations.cpp
)

target_link_libraries(example_basic PRIVATE
    matrixlib
)

add_executable(example_advanced
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/advanced_operations.cpp
)

target_link_libraries(example_advanced PRIVATE
    matrixlib
)

# =============================================================================
# CPU Compile Flags
# =============================================================================

if(MSVC)
    target_compile_options(matrixlib PRIVATE /W4)
    target_compile_options(benchmark_gemm PRIVATE /O2)
else()
    target_compile_options(matrixlib PRIVATE
        -Wall -Wextra -Wpedantic
    )
    target_compile_options(benchmark_gemm PRIVATE
        -O3
    )
endif()

# =============================================================================
# CUDA Section — only when enabled
# =============================================================================

if(OPTI-GEMM_ENABLE_CUDA)

    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # -----------------------------------------------------------------
    # Change this to match your GPU
    #
    #   70 = Volta      (V100)
    #   75 = Turing     (RTX 2080, T4)
    #   80 = Ampere     (A100)
    #   86 = Ampere     (RTX 3060, 3070, 3080)
    #   89 = Ada        (RTX 4090)
    #   90 = Hopper     (H100)
    #
    # Or use native to auto-detect:
    #   set(CMAKE_CUDA_ARCHITECTURES native)
    # -----------------------------------------------------------------

    set(CMAKE_CUDA_ARCHITECTURES native)

    find_package(CUDAToolkit REQUIRED)

    # -----------------------------------------------------------------
    # CUDA GEMM Kernels Library
    # -----------------------------------------------------------------

    add_library(gemm_kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gemm_naive.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gemm_tiled.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gemm_warptile.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/gemm_regblock.cu
    )

    target_include_directories(gemm_kernels PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(gemm_kernels PUBLIC
        CUDA::cudart
    )

    set_target_properties(gemm_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_options(gemm_kernels PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            --ptxas-options=-v
            -lineinfo
        >
    )

    # -----------------------------------------------------------------
    # CUDA Correctness Tests
    # -----------------------------------------------------------------

    add_executable(correctness
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/correctness.cu
    )

    target_link_libraries(correctness PRIVATE
        matrixlib
        gemm_kernels
        CUDA::cudart
    )

    target_include_directories(correctness PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # -----------------------------------------------------------------
    # CUDA Kernel Benchmark
    # -----------------------------------------------------------------

    add_executable(bench_kernels
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_kernels.cu
    )

    target_link_libraries(bench_kernels PRIVATE
        gemm_kernels
        CUDA::cudart
    )

    target_include_directories(bench_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # -----------------------------------------------------------------
    # CUDA vs cuBLAS Benchmark
    # -----------------------------------------------------------------

    add_executable(bench_vs_cublas
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_vs_cublas.cu
    )

    target_link_libraries(bench_vs_cublas PRIVATE
        gemm_kernels
        CUDA::cudart
        CUDA::cublas
    )

    target_include_directories(bench_vs_cublas PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

endif()

# =============================================================================
# Configuration Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== Opti-GEMM Configuration ===")
message(STATUS "  Project version   : ${PROJECT_VERSION}")
message(STATUS "  C++ standard      : ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA enabled      : ${OPTI-GEMM_ENABLE_CUDA}")
if(OPTI-GEMM_ENABLE_CUDA)
    message(STATUS "  CUDA arch         : ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  CUDA toolkit      : ${CUDAToolkit_VERSION}")
endif()
message(STATUS "  Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "===================================")
message(STATUS "")