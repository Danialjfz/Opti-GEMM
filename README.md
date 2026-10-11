# Opti-GEMM

![CUDA](https://img.shields.io/badge/CUDA-11%2B-green)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![License](https://img.shields.io/badge/License-MIT-green)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Danialjfz/Opti-GEMM/blob/develop/notebooks/opti_gemm_colab.ipynb)

> **From-scratch CUDA GEMM optimization journey — exploring the memory hierarchy, GPU architecture, and Nsight profiling on the path toward cuBLAS-level performance.**
>
> Build → Benchmark → Profile → Understand why cuBLAS wins.

***

## Performance Highlights

### Tesla P100:

* Naive CUDA:
  328 GFLOP/s

* Shared Memory GEMM:
  1629 GFLOP/s

≈5x speedup

### Tesla T4:

* Tiled-SMEM:
  441 GFLOP/s

* Register Block:
  1499 GFLOP/s

≈3.4x over tiling, ≈4.5x over naive

***

## Overview

Opti-GEMM is a GPU performance engineering project centered around a single computational kernel: **General Matrix Multiplication (GEMM)**.

The repository starts with a simple CPU implementation used exclusively for correctness validation, then progressively evolves through increasingly optimized CUDA kernels. Rather than treating GPU optimization as a black box, this project treats hardware as a visible playground—empirically measuring how changing memory access patterns and tiling dimensions impacts execution efficiency across different microarchitectures.

***

## Prerequisites

* CUDA toolkit 11.x or newer for GPU builds.

* NVIDIA GPU with supported compute capability for CUDA-enabled targets.

* CMake 3.18+.

* C++17-capable host toolchain (`gcc`, `clang`, or MSVC).

* Nsight Compute installed for profiling with `ncu`.

> Host-only development and CPU benchmarking are supported on macOS, Linux, and Windows. CUDA execution requires an NVIDIA GPU and supported drivers.

## Quickstart

### Build CPU-only targets

```bash
cmake -S . -B build -DOPTI-GEMM_ENABLE_CUDA=OFF
cmake --build build
```

### Build CUDA targets

```bash
cmake -S . -B build -DOPTI-GEMM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;75;86"
cmake --build build -j$(nproc)
```

If your GPU is already installed and supported, you can also use auto-detection:

```bash
cmake -S . -B build -DOPTI-GEMM_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j$(nproc)
```

***

## Run & Profile on a Free Cloud GPU

No NVIDIA GPU locally? The repo ships with a ready-made notebook that builds the
project, verifies correctness, benchmarks the kernels, and profiles them with
**Nsight Compute** — on a free Colab T4 (the same GPU as the benchmark tables below),
or a free Kaggle P100 if you want to reproduce the Pascal numbers:

1. Click the **Open in Colab** badge above.
2. `Runtime → Change runtime type → GPU`, then run all cells.
3. The notebook exports a `.ncu-rep` file you can open in the desktop Nsight Compute
   GUI for rooflines, warp-stall analysis, and source-level metrics.

Profiling workflow, command cheat sheet, and the metrics that matter for GEMM:
[`profiling/nsight_notes.md`](profiling/nsight_notes.md).

***

## Core Objectives

* **Algorithmic Progression:** Implement progressively optimized CUDA GEMM kernels from scratch, transitioning workloads from global VRAM down to shared memory and registers.

* **Architectural Awareness:** Document how distinct NVIDIA architectures (e.g., Pascal vs. Turing) handle unified L1 caching, and why a software optimization that shines on one GPU can be hidden by the hardware on another.

* **Measurement-Driven Engineering:** Profile every iteration using Nsight Compute to trace warp stalls, memory throughput, and register utilization against a production baseline like cuBLAS.

***

## Current Status & Roadmap

CPU Baseline (Stage 0) ──> Naive CUDA (Stage 1) ──> Shared Memory Tiling (Stage 2) ──> Register Blocking (Stage 3) ──> Warp-Level Tiling (Stage 4)

| Optimization Stage                   | Target Layer     | Status      | Engineering Focus                                                                 |
| ------------------------------------ | ---------------- | ----------- | --------------------------------------------------------------------------------- |
| Stage 0: CPU Baseline                | Host System      | Complete    | Matrix allocation pipelines, initialization, and correctness validation.          |
| Stage 1: Naive CUDA                  | Global Memory    | Complete    | Thread/block dimensional alignment, row-major layout, memory coalescing.          |
| Stage 2: Tiled SMEM                  | Shared Memory    | Complete    | Collaborative block-level loading to reduce global VRAM bandwidth pressure.       |
| Stage 3: Reg Block                   | Register File    | Complete    | Thread coarsening via micro-tiles to store intermediate metrics inside registers. |
| Stage 4: Warp Tile                   | Instruction/Warp | In Progress | Explicit warp scheduling layouts and targeting tensor core primitives.            |
| Stage 5: cuBLAS Match and Comparison | Hardware Limit   | Planned     | Profiling handwritten kernels against closed-source assembly-level optimization.  |

> Note: Stage 5 appears as a future planned milestone, while Stages 0-4 represent the primary kernel development path.

## ── Empirical Benchmarks & Hardware Insights

The benchmark suite reveals critical behavioral variances when shifting optimization tiers across different hardware microarchitectures at a fixed  **4096 × 4096 × 4096**  problem size.

### 1. Tesla T4 (Turing Architecture • Compute Capability 7.5)

* **Theoretical Peak:**  \~8,140.80 GFLOP/s (FP32)

* **Architectural Takeaway:**  Turing introduces a unified L1 data cache/shared-memory architecture that can capture temporal locality in global-memory accesses. As a result, naive kernels may benefit from hardware-managed caching, reducing the performance advantage of explicit shared-memory tiling for some workloads.

| Algorithm      | Execution Time (ms) | Attained GFLOP/s | Hardware Efficiency |
| -------------- | ------------------- | ---------------- | ------------------- |
| Naive CUDA     | 415.925             | 330.44           | 4.06%               |
| Tiled-SMEM     | 311.487             | 441.23           | 5.42%               |
| Register Block | 91.700              | 1,498.79         | 18.41%              |

#### Reproduced on a free Colab T4 (size scaling)

Same code, same GPU, via the [Colab notebook](notebooks/opti_gemm_colab.ipynb) — Naive's unified-L1
rescue keeps it ahead of Tiled through 256³, the two tie at 512³, and Tiled pulls ahead from 1024³
onward (+34% at 4096³, where 192 MB of matrices spill the L2). Reg-Block's 16 FLOP/byte block tile
beats both from 256³ onward:

| Problem Size | Naive GFLOP/s | Tiled-SMEM GFLOP/s | Reg-Block GFLOP/s | Winner    |
| ------------ | ------------- | ------------------ | ----------------- | --------- |
| 512³         | 676.79        | 681.37             | 1481.87           | Reg-Block |
| 1024³        | 577.75        | 627.02             | 1603.39           | Reg-Block |
| 2048³        | 424.19        | 469.07             | 1470.10           | Reg-Block |
| 4096³        | 330.44        | 441.23             | 1498.79           | Reg-Block |

> Timings and profiler metrics come from **separate runs** — concurrent `ncu` replay once
> invalidated a row of this table, and the notebook's profiling cells now skip warmups
> deliberately (see [profiling/nsight\_notes.md](profiling/nsight_notes.md)).

#### What Nsight Compute adds to the story (T4, `ncu --section SpeedOfLight`)

Benchmarks alone say *"both kernels sit at \~8% of peak."* Profiling the Naive kernel at 512³
says *why* — and the bottleneck is not where intuition points:

| Metric                  | naive\_gemm @ 512³ | Interpretation                                                                                                  |
| ----------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------- |
| DRAM Throughput         | **1.28%**          | Working set (\~3 MB) fits in the 4 MB L2 — VRAM bandwidth is irrelevant at this size                            |
| L1/TEX Cache Throughput | **88.4%**          | **The real bottleneck** — 2 global loads issued per FMA saturate the on-chip load pipeline                      |
| L2 Cache Throughput     | 4.5%               | L2 absorbs nearly all traffic                                                                                   |
| Compute (SM) Throughput | 80.6%              | Aggregate across *all* SM pipes — the FP32 FMA pipe itself idles (only \~1 of 3 issued instructions is an FFMA) |

**Refined takeaway:** on Turing, "memory-bound" means *L1/issue-bound*, not VRAM-bound —
the Naive kernel is a load-issuing machine that occasionally multiplies. The DRAM story only
appears at 4096³, where 192 MB of matrices blow past the L2. This arithmetic-intensity gap
(0.25 FLOP/byte for Naive) is exactly what Stages 3–4 attack with register and warp tiling.

> Profiling methodology, caveats (clock locking, replay pollution), and the full metric
> cheat sheet: [profiling/nsight\_notes.md](profiling/nsight_notes.md).

***

### 2. Tesla P100 (Pascal Architecture • Compute Capability 6.0)

* **Theoretical Peak:** \~10,608.64 GFLOP/s (FP32 — 56 SMs × 64 FP32 cores/SM × 2 FLOP/FMA × 1.48 GHz boost; PCIe cards ≈ \~9,522 at 1.33 GHz)

* **Architectural Takeaway:** Pascal does not route global memory reads through an L1 data cache by default. Without hardware caching, the Naive kernel is immediately bottlenecked by VRAM latency. By building an L1 cache manually in software via `__shared__` memory, the **Tiled-SMEM kernel yields an instantaneous \~5x speedup**.

| Algorithm  | Execution Time (ms) | Attained GFLOP/s | Hardware Efficiency |
| ---------- | ------------------- | ---------------- | ------------------- |
| Naive CUDA | 417.793             | 328.96           | 3.10%               |
| Tiled-SMEM | 84.319              | 1,629.98         | 15.36%              |

***

### 3. Tesla T4 — Full Benchmark Detail (Turing Architecture • Compute Capability 7.5)

#### Hardware Configuration

The following benchmark was performed on an NVIDIA Tesla T4 GPU.

| Property                  | Specification   |
| ------------------------- | --------------- |
| GPU                       | NVIDIA Tesla T4 |
| Architecture              | Turing          |
| Compute Capability        | 7.5             |
| Streaming Multiprocessors | 40              |
| Precision                 | FP32            |

#### Benchmark Configuration

| Parameter              | Value              |
| ---------------------- | ------------------ |
| Warmup iterations      | 10                 |
| Repetitions per sample | 100                |
| Samples                | 7                  |
| Timing Method          | CUDA Events        |
| Validation             | CPU reference GEMM |

All kernels passed correctness validation for every tested matrix size.

***

#### Performance Results

| Matrix Size |     Naive CUDA |     Tiled-SMEM |  Register Block |
| ----------- | -------------: | -------------: | --------------: |
| 128³        | 229.85 GFLOP/s | 204.80 GFLOP/s |  116.70 GFLOP/s |
| 256³        | 331.86 GFLOP/s | 297.78 GFLOP/s | 1043.25 GFLOP/s |
| 512³        | 676.79 GFLOP/s | 681.37 GFLOP/s | 1481.87 GFLOP/s |
| 1024³       | 577.75 GFLOP/s | 627.02 GFLOP/s | 1603.39 GFLOP/s |
| 2048³       | 424.19 GFLOP/s | 469.07 GFLOP/s | 1470.10 GFLOP/s |
| 4096³       | 330.44 GFLOP/s | 441.23 GFLOP/s | 1498.79 GFLOP/s |

***

#### Architectural Analysis

Unlike Pascal, Turing introduces a unified L1 data cache and shared-memory architecture. This allows simple global-memory kernels to benefit from hardware-managed caching.

As a result, naive GEMM remains competitive for smaller workloads because some memory reuse is captured automatically by the hardware cache hierarchy.

Explicit shared-memory tiling introduces additional operations:

* cooperative tile loading

* synchronization barriers

* shared-memory addressing overhead

For small matrices, these costs can outweigh the benefit of manual data reuse.

However, as matrix dimensions increase, shared-memory tiling becomes increasingly beneficial by reducing redundant global-memory traffic.

At 4096³:

| Kernel         |     Performance | vs. Naive |
| -------------- | --------------: | --------: |
| Naive CUDA     |  330.44 GFLOP/s |     1.00× |
| Tiled-SMEM     |  441.23 GFLOP/s |     1.34× |
| Register Block | 1498.79 GFLOP/s |     4.54× |

The register-blocked kernel achieves approximately **4.5×** the naive implementation and
**3.4×** the tiled kernel at this size.

***

#### Register Blocking Analysis

The register-blocked kernel adds thread-level output reuse on top of
shared-memory tiling:

* Block tile: 64×64
* Threads per block: 16×16 (256 threads, 8 full warps)
* Output per thread: 4×4 (16 accumulators held in registers)

**Why this configuration:** the original 16×16 block tile with a 4×4 thread
block had only 16 threads — half a warp. Every warp executed with 16 of its
32 lanes idle, and the per-SM resident-block limit capped occupancy at a
25% ceiling. The 64×64 block tile with 256 threads removes both problems:
no idle lanes, and arithmetic intensity per block tile rises from 4 to
16 FLOP/byte.

**Measured on the T4** (clean `bench_kernels` run): Reg-Block leads from
256³ onward, peaking at 1,603 GFLOP/s (1024³) and holding ~1.5 TFLOP/s at
4096³ — **4.5×** the naive kernel and **3.4×** the tiled kernel. The only
loss is at 128³, where the 64×64 tile yields a 2×2 grid: four blocks on
40 SMs leave 90% of the GPU idle.

`ptxas -v` confirms the design point: **72 registers per thread** (no
spills) and **33,280 bytes of shared memory per block**. Occupancy is
governed by shared memory, not the block limit: the T4's 64 KB per SM fits
exactly one block, so the kernel runs 8 warps (25% occupancy) and still
wins — 16 independent FMA accumulators per thread hide latency through
instruction-level parallelism instead of thread-level parallelism.

The launch configuration is owned by a host-side launcher
(`regblock_gemm()` in `src/cuda/gemm_regblock.cu`), so callers never pick
grid/block dims and a misconfigured launch is impossible.

Future work: warp-level tiling (Stage 4) and profiling-driven tuning of the
tile parameters.

***

## ── Repository Structure

```Plaintext
Opti-GEMM/
├── CMakeLists.txt               # Unified multi-mode build definition
├── include/
│   ├── MatrixLib.h              # CPU matrix class (correctness oracle)
│   └── cuda/
│       ├── cuda_check.cuh       # Runtime error handling wrapper
│       ├── gemm_naive.cuh       # Stage 1: baseline global-memory kernel
│       ├── gemm_tiled.cuh       # Stage 2: shared-memory tiled kernel
│       ├── gemm_regblock.cuh    # Stage 3: register blocking (host launcher)
│       └── gemm_warptile.cuh    # Stage 4: warp tiling (in progress)
├── src/
│   ├── MatrixLib.cpp            # CPU baseline implementation
│   └── cuda/
│       ├── gemm_naive.cu
│       ├── gemm_tiled.cu
│       ├── gemm_regblock.cu
│       └── gemm_warptile.cu     # stub — writes zeros
├── benchmarks/
│   ├── benchmark_gemm.cpp       # CPU GEMM benchmark
│   ├── bench_kernels.cu         # CUDA kernel benchmark (correctness-gated)
│   └── bench_vs_cublas.cu       # cuBLAS comparison (Stage 5 stub)
├── tests/
│   ├── test_matrix.cpp          # CPU library tests
│   └── correctness.cu           # CUDA kernels vs CPU reference
├── docs/
│   └── kernel_analysis.md       # Per-kernel profiling write-ups (pending)
├── notebooks/
│   └── opti_gemm_colab.ipynb    # One-click build/benchmark/profile on Colab
├── examples/                    # CPU library usage examples (stubs)
└── profiling/
    └── nsight_notes.md          # Profiling workflow, caveats, and metrics
```

## ── Compilation & Build Infrastructure

Opti-GEMM configures automatically based on your system features. It supports zero-hardware editing setups alongside deep native deployment compilation.

### Host-Side Local Build (CPU Only)

For remote engineering, system refactoring, or linting verification on environments lacking localized NVIDIA GPUs (such as macOS workstations):

```bash
cmake -S . -B build -DOPTI-GEMM_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

*Note: this isolates all `.cu` / `.cuh` files from the active runtime targets while maintaining full C++ editor compilation support via your language server.*

### Production Hardware Build (Full CUDA Enablement)

To compile execution binaries for profiling and performance validation on active GPU setups, specify target architectures explicitly or use native auto-detection.

```bash
cmake -S . -B build -DOPTI-GEMM_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="60;75;86"
cmake --build build -j$(nproc)
```

Or if you prefer the local GPU architecture to be detected automatically:

```bash
cmake -S . -B build -DOPTI-GEMM_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j$(nproc)
```

This project sets `CMAKE_CUDA_ARCHITECTURES` to `native` by default inside `CMakeLists.txt`.

### Run targets

After building, use these binaries to verify correctness and measure performance:

```bash
./build/test_matrix            # CPU correctness test
./build/benchmark_gemm         # CPU GEMM benchmark
```

If CUDA is enabled:

```bash
./build/correctness            # CUDA kernel correctness
./build/bench_kernels          # CUDA kernel throughput benchmark
./build/bench_vs_cublas        # CUDA vs cuBLAS comparison
```

### Reproducibility notes

* Use explicit GPU architecture flags when building for fixed hardware.

* Run benchmarks in a stable thermal and power state.

* Collect the output of `nvidia-smi` and `nvcc --version` alongside timing results for comparison.

Verify the low-level physical footprints through the output PTX compiler diagnostics:

```Plaintext
ptxas info    : Compiling entry function 'regblock_gemm_kernel' for 'sm_75'
ptxas info    : Used 72 registers, used 1 barriers, 33280 bytes smem, 388 bytes cmem[0]
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
