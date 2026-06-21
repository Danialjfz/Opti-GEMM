# Opti-GEMM

![CUDA](https://img.shields.io/badge/CUDA-11%2B-green)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **A CUDA-focused GEMM performance engineering project** > Build → Benchmark → Profile → Understand why cuBLAS wins.

---

## Overview

Opti-GEMM is a GPU performance engineering project centered around a single computational kernel: **General Matrix Multiplication (GEMM)**. 

The repository starts with a simple CPU implementation used exclusively for correctness validation, then progressively evolves through increasingly optimized CUDA kernels. Rather than treating GPU optimization as a black box, this project treats hardware as a visible playground—empirically measuring how changing memory access patterns and tiling dimensions impacts execution efficiency across different microarchitectures.

---

## Core Objectives

- **Algorithmic Progression:** Implement progressively optimized CUDA GEMM kernels from scratch, transitioning workloads from global VRAM down to shared memory and registers.
- **Architectural Awareness:** Document how distinct NVIDIA architectures (e.g., Pascal vs. Turing) handle unified L1 caching, and why a software optimization that shines on one GPU can be hidden by the hardware on another.
- **Measurement-Driven Engineering:** Profile every iteration using Nsight Compute to trace warp stalls, memory throughput, and register utilization against a production baseline like cuBLAS.

---

## Current Status & Roadmap

```text
CPU Baseline (Stage 0) ──> Naive CUDA (Stage 1) ──> Shared Memory Tiling (Stage 2) ──> Register Blocking (Stage 3) ──> Warp-Level Tiling (Stage 4)

| Optimization Stage | Target Layer | Status | Engineering Focus |
|-------------------|-------------|--------|------------------|
| Stage 0: CPU Baseline | Host System | Complete | Matrix allocation pipelines, initialization, and correctness validation. |
| Stage 1: Naive CUDA | Global Memory | Complete | Thread/block dimensional alignment, row-major layout, memory coalescing. |
| Stage 2: Tiled SMEM | Shared Memory | Complete | Collaborative block-level loading to reduce global VRAM bandwidth pressure. |
| Stage 3: Reg Block | Register File | In Progress | Thread coarsening via micro-tiles to store intermediate metrics inside registers. |
| Stage 4: Warp Tile | Instruction/Warp | In Progress | Explicit warp scheduling layouts and targeting tensor core primitives. |
| Stage 5: cuBLAS Match | Hardware Limit | Planned | Profiling handwritten kernels against closed-source assembly-level optimization. |

## ── Empirical Benchmarks & Hardware Insights

The benchmark suite reveals critical behavioral variances when shifting optimization tiers across different hardware microarchitectures at a fixed  **4096 × 4096 × 4096**  problem size.


### 1. Tesla T4 (Turing Architecture • Compute Capability 7.5)

-   **Theoretical Peak:**  ~8,140.80 GFLOP/s (FP32)
    
-   **Architectural Takeaway:**  Turing incorporates a highly aggressive, unified L1 Data Cache / Shared Memory layout. The hardware automatically caches global reads for the Naive kernel. Consequently, the manual synchronization barrier (`__syncthreads()`) and pointer arithmetic overhead inside the standard Tiled-SMEM kernel actually run slightly slower than the unoptimized baseline.
    
| Algorithm | Execution Time (ms) | Attained GFLOP/s | Hardware Efficiency |
|-----------|---------------------|------------------|---------------------|
| Naive CUDA | 210.335 | 653.43 | 8.03% |
| Tiled-SMEM | 217.900 | 630.74 | 7.75% |

---
## 2. Tesla P100 (Pascal Architecture • Compute Capability 6.0)

- **Theoretical Peak:** ~19,045.38 GFLOP/s (FP32)  
- **Architectural Takeaway:** Pascal does not route global memory reads through an L1 data cache by default. Without hardware caching, the Naive kernel is immediately bottlenecked by VRAM latency. By building an L1 cache manually in software via `__shared__` memory, the **Tiled-SMEM kernel yields an instantaneous ~5x speedup**.

| Algorithm | Execution Time (ms) | Attained GFLOP/s | Hardware Efficiency |
|-----------|---------------------|------------------|---------------------|
| Naive CUDA | 417.793 | 328.96 | 1.73% |
| Tiled-SMEM | 84.319 | 1,629.98 | 8.56% |

---

## ── Repository Structure



```Plaintext
Opti-GEMM/
├── CMakeLists.txt              # Unified multi-mode build definition
├── include/
│   ├── MatrixLib.h            # Host-side allocation & verification utilities
│   └── cuda/
│       ├── cuda_check.cuh     # Runtime error handling wrappers
│       ├── gemm_naive.cuh     # Baseline global memory execution
│       ├── gemm_tiled.cuh     # Shared memory block-tiled implementations
│       ├── gemm_regblock.cuh  # Thread coarsened layouts (In Progress)
│       └── gemm_warptile.cuh  # Warp synchronized structures (In Progress)
├── src/
│   ├── MatrixLib.cpp          # CPU baseline computations
│   └── cuda/
│       ├── gemm_naive.cu
│       ├── gemm_tiled.cu
│       ├── gemm_regblock.cu
│       └── gemm_warptile.cu
├── benchmarks/
│   ├── benchmark_gemm.cpp     # Host runner entrypoint
│   ├── bench_kernels.cu       # Custom kernel performance matrix
│   └── bench_vs_cublas.cu     # Vendor baseline benchmarking harness
├── tests/
│   └── correctness.cu         # Absolute correctness verification tests
└── profiling/
    └── nsight_notes.md        # Tracked hardware bottlenecks and stall metrics

```

## ── Compilation & Build Infrastructure

Opti-GEMM configures automatically based on your system features. It supports zero-hardware editing setups alongside deep native deployment compilation.

### Host-Side Local Build (CPU Only)

For remote engineering, system refactoring, or linting verification on environments lacking localized NVIDIA GPUs (such as macOS workstations):



```Bash
cmake -S . -B build -DOPTI-GEMM_ENABLE_CUDA=OFF
cmake --build build

```

_Note: This isolates all  `.cu`  /  `.cuh`  files from the active runtime targets while maintaining full C++ editor compilation support via your language server._

### Production Hardware Build (Full CUDA Enablement)

To compile execution binaries for profiling and performance validation on active GPU setups, match the target architecture flags explicitly inside your build sequence (e.g.,  `60`  for Pascal P100,  `75`  for Turing T4,  `86`  for Ampere):



```Bash
cmake -S . -B build -DOPTI-GEMM_ENABLE_CUDA=ON
cmake --build build

```

Verify the low-level physical footprints through the output PTX compiler diagnostics:



```Plaintext
ptxas info : Compiling entry function 'tiled_gemm_kernel' for 'sm_60'
ptxas info : Used 32 registers, 2048 bytes smem, 336 bytes cmem[0]

```

## ── Benchmarking & Profiling Methodology

To capture exact performance characteristics, benchmarks are separated from standard operating system scaling anomalies by following strict validation requirements:

### 1. Clock Stabilization

Prevent hardware performance scaling bias by pinning active GPU core frequencies to their maximum sustained state:



```Bash
sudo nvidia-smi --lock-gpu-clocks=1710,1710

```

### 2. High-Fidelity Timing

All benchmarking runs utilize explicit asynchronous stream synchronization events (`cudaEventRecord`) instead of standard host-side system timers, avoiding CPU driver overhead contamination.

### 3. Deep Metrology via Nsight Compute

Isolate memory bound bottlenecks, compute rooflines, and register bank allocation structures directly from the hardware counter pipelines:



```Bash
ncu --set full -o profiling/tiled_gemm_report ./build/bench_kernels

```

All empirical metrics, architectural performance graphs, and design observations are systematically documented inside  `profiling/nsight_notes.md`.

## ── License

Distributed under the MIT License. See the repository configuration for permissions and scope criteria.
