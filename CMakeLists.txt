  cmake_minimum_required(VERSION 3.10)
project(matrix-engine VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# ========================================
# Library Target
# ========================================
add_library(matrixlib 
    src/MatrixLib.cpp
)

target_include_directories(matrixlib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ========================================
# Test Executable
# ========================================
add_executable(test_matrix tests/test_matrix.cpp)
target_link_libraries(test_matrix matrixlib)

# ========================================
# Benchmark Executable
# ========================================
add_executable(benchmark_gemm benchmarks/benchmark_gemm.cpp)
target_link_libraries(benchmark_gemm matrixlib)

# ========================================
# Example Executables
# ========================================
add_executable(example_basic examples/basic_operations.cpp)
target_link_libraries(example_basic matrixlib)

add_executable(example_advanced examples/advanced_operations.cpp)
target_link_libraries(example_advanced matrixlib)

# ========================================
# Compilation Flags
# ========================================
if(MSVC)
    target_compile_options(matrixlib PRIVATE /W4)
else()
    target_compile_options(matrixlib PRIVATE -Wall -Wextra -Wpedantic)
    # Enable optimizations for benchmark
    target_compile_options(benchmark_gemm PRIVATE -O3)
endif()
