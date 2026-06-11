# Opti-GEMM

![CUDA](https://img.shields.io/badge/CUDA-11%2B-green)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **A CUDA-focused GEMM performance engineering project**  
> Build → Benchmark → Profile → Understand why cuBLAS wins.

---

## Overview

Opti-GEMM is a GPU performance engineering project centered around a single computational kernel:

**General Matrix Multiplication (GEMM)**

The repository starts with a simple CPU implementation used exclusively for correctness validation, then progressively evolves through increasingly optimized CUDA kernels:

- Naive CUDA GEMM
- Shared-memory tiling
- Warp-level tiling
- Register blocking
- cuBLAS benchmarking
- Nsight Compute profiling

The objective is not merely to implement GEMM, but to understand the architectural decisions that make modern GPU libraries exceptionally fast.

> **Core Question:**  
> How does cuBLAS achieve its performance, and how close can a hand-written CUDA kernel get?

---

## Goals

### Primary Objectives

- Implement progressively optimized CUDA GEMM kernels from scratch
- Learn GPU architecture through direct experimentation and measurement
- Benchmark every optimization stage against cuBLAS
- Profile every kernel with Nsight Compute

### Secondary Objectives

- Evolve the best-performing kernel into a small BLAS-style API
- Use CUTLASS as a design reference
- Document the gap between educational kernels and production-grade libraries

### Out of Scope

- CPU SIMD/AVX optimization
- OpenMP multithreading
- PyTorch benchmarking

---

## Current Status

### Implemented

- CPU matrix multiplication baseline
- CUDA project structure
- Naive CUDA GEMM kernel
- CUDA error-checking utilities
- Optional CUDA support through CMake
- `.clangd` configuration for editor tooling
- CPU-only workflow for non-CUDA environments

### In Progress

- Unified CUDA correctness testing
- Benchmark infrastructure
- cuBLAS comparison harness
- Nsight Compute profiling notes

### Planned

- Shared-memory tiled GEMM
- Warp-level tiled GEMM
- Register-blocked GEMM
- Mini-BLAS interface

---

## Why GEMM?

GEMM computes:

```text
C = α·(A × B) + β·C
```

Where:

- `A` → `M × K`
- `B` → `K × N`
- `C` → `M × N`

GEMM is an ideal kernel for performance engineering because it:

- Dominates many HPC and deep learning workloads
- Exercises every level of the GPU memory hierarchy
- Exposes memory access inefficiencies immediately
- Benefits measurably from each optimization stage
- Provides a clear performance baseline through cuBLAS

Studying GEMM naturally leads to understanding:

- Global memory access patterns
- Memory coalescing
- Shared-memory reuse
- Bank conflicts
- Occupancy
- Register pressure
- Arithmetic intensity
- Warp scheduling

---

## Development Workflow

MatrixEngine is designed around a two-machine workflow.

### Development Machine

Typically:

- macOS
- Linux laptop
- Any system without CUDA support

Used for:

- Writing code
- Organizing project structure
- Maintaining the CPU correctness baseline
- Preparing tests and benchmarks
- Validating CPU-only builds

### CUDA Machine

Requirements:

- NVIDIA GPU
- NVIDIA drivers
- CUDA Toolkit
- cuBLAS
- Nsight Compute (recommended)

Used for:

- Building CUDA targets
- Running correctness tests
- Benchmarking kernels
- Performance profiling

---

## Project Structure

```text
MatrixEngine/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── .clangd
│
├── include/
│   ├── MatrixLib.h
│   └── cuda/
│       ├── cuda_check.cuh
│       ├── gemm_naive.cuh
│       ├── gemm_tiled.cuh
│       ├── gemm_warptile.cuh
│       └── gemm_regblock.cuh
│
├── src/
│   ├── MatrixLib.cpp
│   └── cuda/
│       ├── gemm_naive.cu
│       ├── gemm_tiled.cu
│       ├── gemm_warptile.cu
│       └── gemm_regblock.cu
│
├── tests/
│   ├── test_matrix.cpp
│   └── correctness.cu
│
├── benchmarks/
│   ├── benchmark_gemm.cpp
│   ├── bench_kernels.cu
│   └── bench_vs_cublas.cu
│
├── examples/
│   ├── basic_operations.cpp
│   └── advanced_operations.cpp
│
├── profiling/
│   └── nsight_notes.md
│
└── docs/
    └── kernel_analysis.md
```

---

## Build Modes

| Mode | Platform | CUDA Required |
|------|----------|---------------|
| CPU-only | macOS / Linux / Windows | No |
| Full CUDA | Linux / Windows + NVIDIA GPU | Yes |

---

## CPU-Only Build

Recommended for development on systems without CUDA support.

### Requirements

- CMake 3.18+
- C++17 compiler

### Configure

```bash
cmake -S . -B build -DMATRIXENGINE_ENABLE_CUDA=OFF
```

### Build

```bash
cmake --build build
```

### Run

```bash
./build/test_matrix
./build/benchmark_gemm
```

### Notes

- CUDA targets are excluded entirely
- CUDA source files remain editable
- Ideal workflow for macOS development

---

## CUDA Build

Use this mode to build, benchmark, and profile GPU kernels.

### Requirements

- CMake 3.18+
- C++17 compiler
- CUDA Toolkit 11.0+
- NVIDIA GPU
- NVIDIA Driver
- cuBLAS
- Nsight Compute (recommended)

### 1. Verify Installation

```bash
nvcc --version
nvidia-smi
```

### 2. Configure Architecture

Inside `CMakeLists.txt`:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)
```

Common architectures:

| GPU | SM |
|------|------|
| RTX 20 Series | 75 |
| A100 | 80 |
| RTX 30 Series | 86 |
| RTX 40 Series | 89 |
| H100 | 90 |

Or allow CMake to detect automatically:

```cmake
set(CMAKE_CUDA_ARCHITECTURES native)
```

### 3. Configure

```bash
cmake -S . -B build -DMATRIXENGINE_ENABLE_CUDA=ON
```

### 4. Build

```bash
cmake --build build
```

Example PTXAS output:

```text
ptxas info : Compiling entry function 'naive_gemm' for 'sm_86'
ptxas info : Used 16 registers, 0 bytes smem, 336 bytes cmem[0]
```

Useful metrics include:

- Register count
- Shared memory usage
- Constant memory usage

### 5. Run Correctness Tests

```bash
./build/correctness
```

### 6. Run Benchmarks

```bash
./build/bench_kernels
./build/bench_vs_cublas
```

### 7. Profile with Nsight Compute

```bash
ncu --set full -o profiling/naive_gemm_profile ./build/bench_kernels
```

Open the report:

```bash
ncu-ui profiling/naive_gemm_profile.ncu-rep
```

Store observations in:

```text
profiling/nsight_notes.md
```

---

## Reproducible Benchmarking

GPU boost behavior can affect measurements.

### Check Maximum Clock

```bash
nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader
```

### Lock Clocks

```bash
sudo nvidia-smi --lock-gpu-clocks=<min>,<max>
```

Example:

```bash
sudo nvidia-smi --lock-gpu-clocks=1710,1710
```

### Reset

```bash
sudo nvidia-smi --reset-gpu-clocks
```

---

## Example Ubuntu Setup

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential cmake git
```

Verify CUDA:

```bash
nvcc --version
nvidia-smi
```

---

## CUDA Kernel Roadmap

### Stage 0 — CPU Baseline
**Status:** Complete

Simple GEMM implementation used exclusively for correctness validation.

---

### Stage 1 — Naive CUDA GEMM
**Status:** In Progress

One thread computes one output element.

Purpose:

- Establish a performance baseline
- Understand CUDA execution fundamentals
- Measure bandwidth limitations

Key concepts:

- Grid/block/thread indexing
- Row-major memory layout
- Global memory access patterns
- Profiling methodology

---

### Stage 2 — Shared Memory Tiling
**Status:** Planned

Reuse data through cooperative shared-memory loading.

### Stage 3 — Warp-Level Tiling
**Status:** Planned

Assign larger output tiles to warp-level execution units.

### Stage 4 — Register Blocking
**Status:** Planned

Increase arithmetic intensity by accumulating multiple outputs per thread.

### Stage 5 — cuBLAS Comparison
**Status:** Planned

Benchmark every kernel version against cuBLAS.

### Stage 6 — Mini-BLAS API
**Status:** Planned

Expose the best implementation through a reusable interface.

---

## Benchmarking Methodology

All performance work is measurement-driven.

### Timing

Use CUDA events rather than CPU-side timers.

### Protocol

- Warm-up iterations
- Multiple timed runs
- Correctness validation before benchmarking
- Optional GPU clock locking

### Metrics

- Runtime
- GFLOP/s
- Speedup versus previous version
- Percentage of cuBLAS performance
- Occupancy
- Memory throughput
- Register usage
- Warp stall reasons

---

## Profiling Philosophy

Every optimization step should be justified by profiling data.

Key metrics:

- Achieved occupancy
- Global memory throughput
- Shared memory throughput
- Bank conflicts
- Register count
- Arithmetic intensity
- FFMA utilization
- Warp stall reasons

All findings belong in:

```text
profiling/nsight_notes.md
```

---

## GPU Memory Hierarchy

```text
Register File    ~1 cycle
Shared Memory    ~5 cycles
L1 Cache         ~30 cycles
L2 Cache         ~100–200 cycles
Global Memory    ~300–600 cycles
```

Every optimization in this project is fundamentally about moving data closer to registers and maximizing reuse before returning to global memory.

---

## Tooling

### .clangd

Included to improve CUDA editor support on non-CUDA systems.

Benefits:

- Better IntelliSense
- Fewer false diagnostics
- Improved `.cu` and `.cuh` support

### compile_commands.json

Enable via:

```cmake
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
```

Useful for:

- clangd
- Language servers
- IDE tooling

---

## Long-Term Vision

```text
CPU Baseline
      ↓
Naive CUDA GEMM
      ↓
Shared-Memory Tiling
      ↓
Warp-Level Tiling
      ↓
Register Blocking
      ↓
cuBLAS Comparison
      ↓
Mini-BLAS Library
```

The objective is not to outperform cuBLAS.

The objective is to understand, measure, and document why it performs so well.

---

## License

MIT License
