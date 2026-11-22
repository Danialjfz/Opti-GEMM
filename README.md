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

### Tesla T4 (vs cuBLAS, FP32 pedantic math):

* Tiled-SMEM:
  456 GFLOP/s (0.13x cuBLAS)

* Register Block:
  1499 GFLOP/s (0.44x cuBLAS)

* Warp-Tile:
  2443 GFLOP/s (0.72x cuBLAS)

≈6.6x over naive, ≈5.4x over tiling at 4096³ — cuBLAS reaches 3415 GFLOP/s

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
| Stage 4: Warp Tile                   | Instruction/Warp | Complete    | Three-level block→warp→thread hierarchy; 128×128 block tile, 8×8 micro-tile, 64 accumulators/thread. |
| Stage 5: cuBLAS Match and Comparison | Hardware Limit   | In Progress | `bench_vs_cublas` validates and times every kernel against cuBLAS (pedantic FP32) across 11 shapes; warp-tile peaks at 0.79x cuBLAS. |

> Note: Stages 0-4 are the completed kernel development path; Stage 5 (closing the remaining
> gap to cuBLAS) is the active milestone, with the comparison harness already in place.

## ── Empirical Benchmarks & Hardware Insights

The benchmark suite reveals critical behavioral variances when shifting optimization tiers across different hardware microarchitectures at a fixed  **4096 × 4096 × 4096**  problem size.

### 1. Tesla T4 (Turing Architecture • Compute Capability 7.5)

* **Theoretical Peak:**  \~8,140.80 GFLOP/s (FP32)

* **Architectural Takeaway:**  Turing introduces a unified L1 data cache/shared-memory architecture that can capture temporal locality in global-memory accesses. As a result, naive kernels may benefit from hardware-managed caching, reducing the performance advantage of explicit shared-memory tiling for some workloads.

| Algorithm      | Execution Time (ms) | Attained GFLOP/s | Hardware Efficiency | vs cuBLAS |
| -------------- | ------------------- | ---------------- | ------------------- | --------- |
| Naive CUDA     | 370.772             | 370.68           | 4.55%               | 0.11x     |
| Tiled-SMEM     | 301.382             | 456.03           | 5.60%               | 0.13x     |
| Register Block | 91.709              | 1,498.65         | 18.41%              | 0.44x     |
| Warp-Tile      | 56.266              | 2,442.65         | 30.00%              | 0.72x     |
| cuBLAS         | 40.250              | 3,414.63         | 41.94%              | 1.00x     |

> Hardware efficiency is relative to the T4's ~8,140.8 GFLOP/s FP32 theoretical peak.

#### Reproduced on a free Colab T4 (size scaling, handwritten kernels vs cuBLAS)

Same code, same GPU, via the [Colab notebook](notebooks/opti_gemm_colab.ipynb), measured by
`bench_vs_cublas` (warmup 10, 100 repeats/sample, 7 samples, median reported; cuBLAS in
`CUBLAS_PEDANTIC_MATH` FP32 mode). Naive's unified-L1 rescue keeps it ahead of Tiled only at
128³; Tiled pulls ahead from 256³ onward. Reg-Block's 16 FLOP/byte block tile dominates the
small-to-mid range, and Warp-Tile's 32 FLOP/byte hierarchy takes over from 512³, holding
0.70–0.76x of cuBLAS at large sizes:

| Problem Size | Naive GFLOP/s | Tiled-SMEM GFLOP/s | Reg-Block GFLOP/s | Warp-Tile GFLOP/s | cuBLAS GFLOP/s | Winner    |
| ------------ | ------------- | ------------------ | ----------------- | ----------------- | -------------- | --------- |
| 128³         | 323.42        | 287.53             | 169.43            | 53.69             | 477.46         | Naive     |
| 256³         | 499.97        | 563.02             | 1045.45           | 363.38            | 1755.82        | Reg-Block |
| 512³         | 668.06        | 674.90             | 1476.75           | 1598.24           | 5381.66        | Warp-Tile |
| 1024³        | 583.65        | 626.21             | 1597.81           | 2515.96           | 3611.88        | Warp-Tile |
| 2048³        | 463.25        | 503.16             | 1555.87           | 2607.06           | 3423.54        | Warp-Tile |
| 4096³        | 370.68        | 456.03             | 1498.65           | 2442.65           | 3414.63        | Warp-Tile |

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

| Matrix Size        |     Naive CUDA |     Tiled-SMEM |  Register Block |      Warp-Tile |         cuBLAS | Warp-Tile vs cuBLAS |
| ------------------ | -------------: | -------------: | --------------: | -------------: | -------------: | ------------------: |
| 128³               | 323.42 GFLOP/s | 287.53 GFLOP/s |  169.43 GFLOP/s |  53.69 GFLOP/s |  477.46 GFLOP/s |               0.11x |
| 256³               | 499.97 GFLOP/s | 563.02 GFLOP/s | 1045.45 GFLOP/s | 363.38 GFLOP/s | 1755.82 GFLOP/s |               0.21x |
| 512³               | 668.06 GFLOP/s | 674.90 GFLOP/s | 1476.75 GFLOP/s | 1598.24 GFLOP/s | 5381.66 GFLOP/s |               0.30x |
| 1024³              | 583.65 GFLOP/s | 626.21 GFLOP/s | 1597.81 GFLOP/s | 2515.96 GFLOP/s | 3611.88 GFLOP/s |               0.70x |
| 2048³              | 463.25 GFLOP/s | 503.16 GFLOP/s | 1555.87 GFLOP/s | 2607.06 GFLOP/s | 3423.54 GFLOP/s |               0.76x |
| 4096³              | 370.68 GFLOP/s | 456.03 GFLOP/s | 1498.65 GFLOP/s | 2442.65 GFLOP/s | 3414.63 GFLOP/s |               0.72x |
| 1000³              | 517.19 GFLOP/s | 560.88 GFLOP/s | 1317.51 GFLOP/s | 2144.39 GFLOP/s | 2867.43 GFLOP/s |               0.75x |
| 1023 × 1025 × 1027 | 458.70 GFLOP/s | 529.17 GFLOP/s | 1306.40 GFLOP/s | 2091.71 GFLOP/s | 2639.15 GFLOP/s |               0.79x |
| 4096 × 128 × 1024  | 533.16 GFLOP/s | 584.92 GFLOP/s | 1279.25 GFLOP/s | 1891.36 GFLOP/s | 2530.92 GFLOP/s |               0.75x |
| 128 × 4096 × 1024  | 386.73 GFLOP/s | 462.31 GFLOP/s | 1260.47 GFLOP/s | 1963.35 GFLOP/s | 2879.55 GFLOP/s |               0.68x |
| 1024 × 1024 × 32   | 526.50 GFLOP/s | 530.99 GFLOP/s |  647.96 GFLOP/s | 779.44 GFLOP/s | 2315.76 GFLOP/s |               0.34x |

All five implementations pass correctness validation against the double-precision-accumulating
CPU reference at every shape (worst handwritten-kernel max abs error: 3.51e-04 at 4096³). The
odd and non-square shapes (rows 7–11) exercise the bounds-checked edge paths of every kernel;
Warp-Tile's best relative result (0.79x) comes from the fully irregular 1023 × 1025 × 1027 case.

***

#### Cross-Validation on a Second T4 (Nsight Systems Timeline, Kaggle)

The full `bench_kernels` suite was also captured under `nsys` on a different
Tesla T4 (Kaggle, shared VM) and reconstructed from the CUPTI kernel trace:
17,064 launches, exactly 4 kernels × 6 sizes × (1 validation + 10 warmup +
700 timed). This is an independent measurement path — different GPU,
different host, different timing mechanism (CUPTI kernel durations instead
of `cudaEvent` medians):

| Kernel @4096³ | Colab T4 (`cudaEvent`) | Kaggle T4 (CUPTI trace) |   Δ |
| ------------- | ---------------------: | ----------------------: | --: |
| Naive CUDA    |              370.77 ms |               358.93 ms | −3% |
| Tiled-SMEM    |              301.38 ms |               289.32 ms | −4% |
| Register Block|               91.71 ms |                87.46 ms | −5% |
| Warp-Tile     |               56.27 ms |                53.98 ms | −4% |

Warp-Tile at 2048³ agrees to within **0.1%** (6.590 ms vs 6.598 ms). The
trace also confirms the launch configuration reported by `ptxas`: 72
registers for Reg-Block and 111 for Warp-Tile (2 resident blocks per SM, as
designed). Reproduce with `nsys profile -o timeline ./build/bench_kernels`
and `nsys export --type sqlite`, querying `CUPTI_ACTIVITY_KIND_KERNEL`.

Three honest caveats the trace makes visible:

* **Thermal drift on shared, passively cooled T4s.** The DRAM-bound Naive
  kernel slows ~10% between its first and last 50 launches at 2048³ and
  4096³ (70 W power envelope). All tables here report *sustained* medians;
  cold first launches run faster — Warp-Tile's coldest single 4096³ launch
  reached 3.7 TFLOP/s vs ~2.5 TFLOP/s sustained. Treat minimums as a boost
  ceiling, never as the number to cite.
* **128³–256³ rows are launch-latency-bound.** At 13–60 µs per kernel the
  GPU idles between launches on a shared vCPU; treat ±20% deltas at these
  sizes as noise, not kernel quality.
* **The CPU reference dominates wall time on shared vCPUs.** The 4096³
  reference alone consumed ~12 of 24 minutes in the Kaggle run (GPU busy:
  44% of the trace). For repeat runs, cache the reference or skip
  validation above 2048³.

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

| Kernel         |     Performance | vs. Naive | vs cuBLAS |
| -------------- | --------------: | --------: | --------: |
| Naive CUDA     |  370.68 GFLOP/s |     1.00× |     0.11× |
| Tiled-SMEM     |  456.03 GFLOP/s |     1.23× |     0.13× |
| Register Block | 1498.65 GFLOP/s |     4.04× |     0.44× |
| Warp-Tile      | 2442.65 GFLOP/s |     6.59× |     0.72× |
| cuBLAS         | 3414.63 GFLOP/s |     9.21× |     1.00× |

The warp-tiled kernel achieves approximately **6.6×** the naive implementation,
**5.4×** the tiled kernel, and **1.6×** the register-blocked kernel at this size —
closing to within **1.4×** of cuBLAS in full FP32 mode.

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

**Measured on the T4** (`bench_vs_cublas`): Reg-Block leads from
256³ through 512³, peaking at 1,598 GFLOP/s (1024³) and holding ~1.5 TFLOP/s at
4096³ — **4.0×** the naive kernel and **3.3×** the tiled kernel. Its weak
spot is 128³, where the 64×64 tile yields a 2×2 grid: four blocks on
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

***

#### Warp-Tile Analysis

The warp-tiled kernel (`src/cuda/gemm_warptile.cu`) adds an explicit warp
level between the block and the thread, giving a three-level hierarchy:

* Block tile: 128×128
* Threads per block: 256 (8 full warps, arranged 4 warps × 2 warps)
* Warp tile: 32×64
* Output per thread: 8×8 (64 accumulators held in registers)
* K-tile: 8, with the `As` shared tile padded to 128×9 to kill bank conflicts

**Why this configuration:** Reg-Block's 64×64 block tile amortizes each
global load over 16 FLOP/byte. Doubling the block tile to 128×128 raises
that to 32 FLOP/byte, and the 8×8 micro-tile gives each thread 64
independent FMAs per shared-memory load (vs 16 in Reg-Block) — enough ILP
to keep the FP32 pipes busy at deliberately low occupancy. The 4×2 warp
arrangement keeps adjacent warps on the same A rows for L2-friendly B
reuse. Shared memory drops to ~8.7 KB per block, leaving headroom for a
double-buffered second tile in a later stage.

**Measured on the T4** (`bench_vs_cublas`): Warp-Tile takes the handwritten
lead from 512³ onward — 2,516 GFLOP/s at 1024³, a peak of **2,607 GFLOP/s
at 2048³ (0.76× cuBLAS)**, and 2,443 GFLOP/s at 4096³ (**6.6×** naive,
**5.4×** tiled, **1.6×** Reg-Block). Its best relative result is **0.79×
cuBLAS** at the fully irregular 1023 × 1025 × 1027 shape, showing the
bounds-checked edge handling costs almost nothing. The weaknesses are at
the extremes: at 128³ the 128×128 block tile produces a single thread
block for 40 SMs (0.11× cuBLAS), and on the skinny 1024 × 1024 × 32 case
(K = 32 → four BK phases) load/store overhead dominates (0.34× cuBLAS).

**Remaining gap to cuBLAS:** cuBLAS peaks at 5,382 GFLOP/s (512³, ~66% of
the T4's FP32 theoretical peak) and sustains ~3.4 TFLOP/s at large square
sizes — roughly 1.4× ahead of Warp-Tile there. The known next
optimizations that close this gap are vectorized `float4` global/shared
loads and shared-memory double buffering, both of which the three-level
hierarchy was structured to accommodate.

The launch configuration is owned by a host-side launcher
(`warptile_gemm()`), matching the Reg-Block design.

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
│       └── gemm_warptile.cuh    # Stage 4: warp tiling interface
├── src/
│   ├── MatrixLib.cpp            # CPU baseline implementation
│   └── cuda/
│       ├── gemm_naive.cu
│       ├── gemm_tiled.cu
│       ├── gemm_regblock.cu
│       └── gemm_warptile.cu     # Stage 4: warp-tiled CUDA kernel
├── benchmarks/
│   ├── benchmark_gemm.cpp       # CPU GEMM benchmark
│   ├── bench_kernels.cu         # CUDA kernel benchmark (correctness-gated)
│   └── bench_vs_cublas.cu       # CUDA kernels vs. cuBLAS benchmark
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

`bench_vs_cublas` validates every handwritten kernel and cuBLAS result against
the CPU reference before timing. It uses row-major host/device buffers and
converts the cuBLAS call to the equivalent column-major `BᵀAᵀ` operation.
cuBLAS is configured with `CUBLAS_PEDANTIC_MATH` so the comparison uses full
FP32 arithmetic rather than reduced-precision tensor-core math. The comparison
includes square, odd, and non-square shapes; all bundled kernels use bounds
checks for arbitrary M, N, and K.

### Reproducibility notes

* Use explicit GPU architecture flags when building for fixed hardware.

* Run benchmarks in a stable thermal and power state.

* Collect GPU clocks, power, and temperature while the benchmark runs:

  ```bash
  nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv -l 1
  ```

* Repeat the complete suite two or three times to reduce ordering and thermal
  bias. Small GEMMs are warm-L2 measurements because inputs remain resident
  between samples; use a cache-control experiment separately for cold-cache
  measurements.

* On shared cloud VMs (Colab, Kaggle), expect ±5% variation from neighbors
  and clock drift, plus ~10% thermal sag on long DRAM-bound kernels. The
  tables above were reproduced on two different T4s within that band (see
  the cross-validation table); sustained medians are the citable numbers.

* Budget wall time for the CPU reference GEMM on shared vCPUs — the 4096³
  reference alone costs ~12 minutes. Cache or skip it on repeat runs.

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
