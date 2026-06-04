# MatrixEngine

A modern C++ matrix library and performance engineering project focused on understanding how high-performance linear algebra systems are built.

The goal of this repository is not simply to implement matrix operations, but to study the complete optimization journey from a naive CPU implementation to production-grade matrix multiplication techniques used in modern machine learning and high-performance computing systems.

The project begins with a custom matrix library and progressively explores cache-aware algorithms, SIMD vectorization, multithreading, CUDA kernels, and comparisons against industry-standard libraries such as cuBLAS, CUTLASS, and PyTorch.

---

## Project Goals

This repository is designed as a hands-on exploration of:

- Modern C++ systems programming
- Dynamic memory management
- Cache-aware algorithm design
- Linear algebra kernels (GEMM)
- CPU performance optimization
- GPU programming with CUDA
- Benchmarking methodology
- High Performance Computing (HPC)
- Machine Learning infrastructure

The ultimate objective is to understand why highly optimized libraries such as cuBLAS and CUTLASS are fast and to quantify the performance gap between educational implementations and production-grade systems.

---

## Current Features

### Matrix Core

- Dynamic matrix allocation
- Contiguous row-major memory layout
- Bounds-checked element access
- Deep-copy semantics
- Move semantics
- Rule of Five implementation
- Matrix resizing

### Matrix Operations

- Matrix addition
- Matrix subtraction
- Scalar multiplication
- Matrix transpose
- Matrix multiplication (GEMM)

### Benchmarking Infrastructure

- High-resolution timing
- Multiple benchmark iterations
- Warmup runs
- Optimization-resistant benchmark sink
- Reproducible benchmark framework

---

## Why GEMM?

General Matrix Multiplication (GEMM) is one of the most important kernels in computing.

Many workloads ultimately reduce to matrix multiplication:

- Deep learning
- Scientific computing
- Computer vision
- Signal processing
- Numerical simulation

Because of this, a significant amount of engineering effort is devoted to making GEMM as efficient as possible.

This repository focuses heavily on understanding GEMM performance across CPU and GPU architectures.

---

## Memory Layout

Matrices are stored using contiguous row-major memory.

Example:

```text
1 2 3
4 5 6
```

is stored as:

```text
[1][2][3][4][5][6]
```

Element access:

```cpp
data[row * columns + column]
```

Understanding memory layout is critical because modern processor performance is often limited by memory access patterns rather than arithmetic throughput.

---

## Example Usage

```cpp
MatrixLib A(2, 2);

A.at(0,0) = 1;
A.at(0,1) = 2;
A.at(1,0) = 3;
A.at(1,1) = 4;

MatrixLib B = A.transpose();

B.print();
```

Output:

```text
1 3
2 4
```

---

## Project Structure

```text
MatrixEngine/
│
├── include/
│   └── MatrixLib.h
│
├── src/
│   └── MatrixLib.cpp
│
├── tests/
│   └── main.cpp
│
├── benchmarks/
│   └── benchmark_gemm.cpp
│
├── CMakeLists.txt
│
└── README.md
```

---

## Building

### Requirements

- CMake 3.16+
- C++17 compatible compiler

Supported compilers:

- GCC
- Clang
- MSVC

### Build

```bash
mkdir build
cd build

cmake ..
cmake --build .
```

---

## Running Tests

```bash
./matrix_tests
```

---

## Running Benchmarks

```bash
./matrix_benchmark
```

---

# Development Roadmap

The repository is intentionally developed in stages.

Each stage explores a new optimization technique and measures its impact.

---

## Stage 1 — Matrix Library Fundamentals

**Status:** Complete

- Dynamic memory management
- Rule of Five
- Matrix arithmetic
- Naive GEMM
- Benchmarking infrastructure

### Concepts Studied

- RAII
- Move semantics
- Memory ownership
- Row-major layout

---

## Stage 2 — Cache-Aware GEMM

**Status:** Planned

### Objectives

- Implement tiled matrix multiplication
- Improve cache locality
- Reduce cache misses
- Benchmark against naive GEMM

### Concepts Studied

- CPU cache hierarchy
- Data reuse
- Blocking strategies
- Memory access optimization

---

## Stage 3 — SIMD Optimization

**Status:** Planned

### Objectives

- AVX2 implementation
- AVX-512 implementation (where available)
- Compare vectorized and scalar execution

### Concepts Studied

- Vector registers
- Instruction-level parallelism
- Throughput optimization

---

## Stage 4 — Multithreaded CPU GEMM

**Status:** Planned

### Objectives

- OpenMP parallelization
- Thread scaling experiments
- Multi-core utilization

### Concepts Studied

- Parallel programming
- CPU scalability
- Memory bandwidth limitations

---

## Stage 5 — CUDA GEMM

**Status:** Planned

### Objectives

- Basic CUDA matrix multiplication kernel
- One thread per output element
- CPU vs GPU comparison

### Concepts Studied

- CUDA programming model
- Thread hierarchy
- GPU execution model

---

## Stage 6 — Shared-Memory CUDA GEMM

**Status:** Planned

### Objectives

- CUDA tiling
- Shared memory optimization
- Occupancy analysis
- Performance tuning

### Concepts Studied

- Shared memory
- Memory coalescing
- Block-level optimization

---

## Stage 7 — Production Library Comparisons

**Status:** Planned

### Benchmarks Against

- cuBLAS
- CUTLASS
- PyTorch
- CPU implementation

### Metrics

- Execution time
- GFLOP/s
- Scaling behavior
- Efficiency relative to optimized libraries

---

## Benchmarking Methodology

Future benchmarks will report:

- Average runtime
- Minimum runtime
- Maximum runtime
- Standard deviation
- GFLOP/s
- Relative speedup

Benchmark sizes will include:

```text
256 x 256
512 x 512
1024 x 1024
2048 x 2048
4096 x 4096
```

The goal is to measure optimization impact quantitatively rather than relying on intuition.

---

## Learning Outcomes

This project explores:

- Modern C++
- Systems programming
- Performance engineering
- Numerical computing
- CUDA programming
- HPC concepts
- Benchmark design
- Machine learning infrastructure

---

## Long-Term Vision

By the final stage, this repository will document the complete evolution of a matrix multiplication engine:

```text
Naive GEMM
      ↓
Cache-Aware GEMM
      ↓
SIMD GEMM
      ↓
OpenMP GEMM
      ↓
CUDA GEMM
      ↓
Shared-Memory CUDA GEMM
      ↓
CUTLASS / cuBLAS Comparison
```

The purpose is not to outperform industrial libraries, but to understand the engineering decisions that make them fast.

---

## License

MIT License