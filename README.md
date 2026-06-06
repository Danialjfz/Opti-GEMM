# MatrixEngine

> **A CUDA-focused GEMM performance engineering project.**
> Build. Profile. Measure. Understand why cuBLAS wins.

---

## What This Is

MatrixEngine is a GPU performance engineering study built around one kernel: **General Matrix Multiplication (GEMM)**.

The project starts with the most naive CUDA implementation possible — one thread per output element, no shared memory, no optimization — and evolves it through every major GPU optimization technique, benchmarked against cuBLAS at every stage.

A CPU implementation exists as a **correctness oracle only**. It is not optimized. It is not a performance target. It exists to verify that CUDA kernels produce correct results before timing begins.

The central question this project answers:

> *Why is cuBLAS fast, and how close can a hand-written CUDA kernel get?*

---

## What This Is Not

- A general-purpose C++ matrix library
- A CPU optimization study
- A production linear algebra system
- An attempt to outperform cuBLAS

The CPU code is deliberately naive and will stay that way. Every engineering decision in this project is made in service of understanding GPU architecture — not C++ performance.

---

## Why GEMM?

GEMM computes:

```
C = α·(A×B) + β·C
```

where **A** is M×K, **B** is K×N, and **C** is M×N.

It is the right kernel to study because:

- It dominates runtime in deep learning, scientific computing, and signal processing
- It exercises **every level of the GPU memory hierarchy**
- It transitions from memory-bound to compute-bound as matrix size grows
- Every major optimization technique — tiling, coalescing, occupancy tuning, register blocking — has **measurable, quantifiable impact**
- cuBLAS and CUTLASS expose their design decisions through their performance characteristics

Understanding GEMM performance means understanding GPU architecture. There is no shortcut.

---

## Project Goals

### Primary
- Build progressively optimized CUDA GEMM kernels from scratch
- Profile every kernel with **Nsight Compute** before moving forward
- Benchmark every stage against **cuBLAS** — this is the scorecard
- Understand the GPU memory hierarchy through direct measurement

### Secondary
- Evolve the best kernel into a structured **mini-BLAS-style library**
- Study CUTLASS as an architectural reference for kernel design decisions
- Document the performance gap between educational and production implementations with **real numbers**

### Explicitly Out of Scope
- SIMD / AVX CPU optimization
- OpenMP multithreading
- PyTorch comparison

---

## GPU Memory Hierarchy

Every optimization in this project is fundamentally about moving data **closer to the registers** and keeping it there as long as possible.

```
─────────────────────────────────────────────────────────────
  Register File     ~8 KB per SM          ~1 cycle latency
─────────────────────────────────────────────────────────────
  Shared Memory     ~48–96 KB per SM      ~5 cycle latency
─────────────────────────────────────────────────────────────
  L1 Cache          ~32–128 KB per SM     ~30 cycle latency
─────────────────────────────────────────────────────────────
  L2 Cache          ~4–40 MB              ~100–200 cycle latency
─────────────────────────────────────────────────────────────
  Global Memory     ~8–80 GB              ~300–600 cycle latency
─────────────────────────────────────────────────────────────
```

The naive kernel lives entirely in global memory.
Every optimization stage moves work up this hierarchy.

---

## Project Structure

```
MatrixEngine/
│
├── include/
│   ├── MatrixLib.h                  # CPU matrix — correctness baseline only
│   └── cuda/
│       ├── gemm_naive.cuh           # Kernel v1: one thread per output element
│       ├── gemm_tiled.cuh           # Kernel v2: shared memory tiling
│       ├── gemm_warptile.cuh        # Kernel v3: warp-level tiling
│       └── gemm_regblock.cuh        # Kernel v4: register blocking
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
│   └── correctness.cu               # All kernels verified against CPU baseline
│
├── benchmarks/
│   ├── bench_kernels.cu             # All CUDA kernels head-to-head
│   └── bench_vs_cublas.cu           # Your kernels vs cuBLAS
│
├── profiling/
│   └── nsight_notes.md              # Nsight Compute findings per kernel
│
├── docs/
│   └── kernel_analysis.md           # Architecture decisions, tradeoffs, numbers
│
├── CMakeLists.txt
└── README.md
```

---

## Roadmap

Each stage has one rule: **benchmark against cuBLAS before moving forward.**

The gap between your kernel and cuBLAS peak is the only scorecard that matters.

---

### Stage 0 — CPU Baseline
**Status: `Complete`**

A deliberately naive CPU GEMM implementation. Triple-loop. No optimization. Exists purely to verify GPU kernel correctness.

**Concepts established:**
- RAII, Rule of Five, move semantics
- Row-major memory layout and why it matters
- Benchmark harness design

**This code will not be touched again.**

---

### Stage 1 — Naive CUDA GEMM
**Status: `In Progress`**

One thread computes one output element. Every thread independently walks the entire K dimension. All memory reads come from global memory.

```
Thread (tx, ty)  →  computes C[ty][tx]
Each thread reads full row of A and full column of B from global memory
No cooperation between threads
No data reuse
```

**Expected outcome:**
- Correct results, terrible performance
- Bandwidth-limited, uncoalesced memory access
- Establishes the performance floor

**Profile with Nsight Compute:**
- Global memory throughput
- Memory access pattern (coalesced vs. strided)
- L2 hit rate
- Achieved occupancy

**Benchmark report format:**

| Kernel | 512² | 1024² | 2048² | 4096² |
|---|---|---|---|---|
| Naive CUDA | X GFLOP/s | X GFLOP/s | X GFLOP/s | X GFLOP/s |
| cuBLAS | X GFLOP/s | X GFLOP/s | X GFLOP/s | X GFLOP/s |
| % of Peak | X% | X% | X% | X% |

**Concepts studied:**
- CUDA thread hierarchy — thread, warp, block, grid
- What memory coalescing means in hardware terms
- Why naive GEMM is bandwidth-limited, not compute-limited
- How to read Nsight Compute output

---

### Stage 2 — Shared Memory Tiled GEMM
**Status: `Planned`**

Threads within a block cooperate to load tiles of A and B into shared memory. Each tile is reused by all threads in the block before the next tile is loaded.

```
Block loads TILE_SIZE × TILE_SIZE region of A → shared memory
Block loads TILE_SIZE × TILE_SIZE region of B → shared memory
All threads compute partial products from shared memory
Slide tiles across K dimension, accumulate into registers
Write final result to global memory
```

**Expected outcome:**
- Significant reduction in global memory transactions
- Measurable GFLOP/s improvement over Stage 1
- Shared memory bank conflicts become the next bottleneck

**Profile with Nsight Compute:**
- Shared memory bank conflicts
- L1 / shared memory throughput vs. Stage 1
- Arithmetic intensity
- Memory stall cycles vs. Stage 1

**Concepts studied:**
- Shared memory as programmer-controlled L1
- Tile size selection and its effect on occupancy
- Bank conflict patterns and avoidance
- The relationship between tile size and register pressure

---

### Stage 3 — Warp-Level Tiling
**Status: `Planned`**

Each warp collectively owns a larger output tile. Threads within the warp handle non-overlapping sub-regions, reducing redundant loads and increasing arithmetic intensity per memory transaction.

**Expected outcome:**
- Improved data reuse at the warp level
- Higher GFLOP/s from reduced memory pressure
- Register pressure increases — occupancy may decrease

**Profile with Nsight Compute:**
- Warp efficiency
- Register file usage
- Occupancy vs. register pressure tradeoff
- Warp stall reasons

**Concepts studied:**
- Warp scheduling and latency hiding
- How occupancy affects instruction throughput
- Register pressure as a first-class hardware resource
- The occupancy vs. ILP tradeoff

---

### Stage 4 — Register Blocking
**Status: `Planned`**

Each thread accumulates multiple output values entirely in registers. The inner computation loop is restructured so that arithmetic density per memory load is maximized.

```
Thread computes TM × TN tile of C
Outer loop over K tiles
Inner loops over TM and TN — all accumulation in registers
Shared memory reads are amortized across many FFMA instructions
```

**Expected outcome:**
- Highest GFLOP/s of all hand-written kernels
- Closest approach to cuBLAS peak
- Register file spilling becomes a real concern if tile size is too large

**Profile with Nsight Compute:**
- Register spilling to local memory
- FFMA instruction ratio
- Achieved vs. theoretical FLOP/s
- Roofline position: memory-bound or compute-bound?

**Concepts studied:**
- Register file as the fastest memory tier
- How to maximize FFMA utilization
- When register blocking hurts (spilling to local memory)
- How CUTLASS approaches this problem at scale

---

### Stage 5 — cuBLAS Analysis and Gap Report
**Status: `Planned`**

Integrate cuBLAS SGEMM and produce a full, systematic performance gap analysis across all kernel versions and all matrix sizes.

**Full benchmark table:**

| Kernel | 256² | 512² | 1024² | 2048² | 4096² |
|---|---|---|---|---|---|
| Naive CUDA | — | — | — | — | — |
| Shared Memory | — | — | — | — | — |
| Warp Tiled | — | — | — | — | — |
| Register Blocked | — | — | — | — | — |
| **cuBLAS** | — | — | — | — | — |
| **% of cuBLAS Peak** | — | — | — | — | — |

**What the gap analysis will reveal:**
- cuBLAS selects between many specialized kernel variants at runtime
- It applies double buffering and software pipelining to hide memory latency
- It uses Tensor Cores where available
- The gap is real, documented, and explainable — not just a number

**Concepts studied:**
- How to profile cuBLAS with Nsight Compute
- What software pipelining and double buffering buy you
- Why architecture-specific kernels matter
- How to read the roofline model for your GPU

---

### Stage 6 — Mini-BLAS Library
**Status: `Planned`**

Evolve the best-performing kernel into a structured, reusable library with a clean API.

```cpp
matengine_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
```

**Planned features:**
- Non-square matrix support
- Configurable tile sizes via template parameters
- Batch GEMM support
- FP16 / half-precision variant

**Design reference:** CUTLASS kernel hierarchy — studied as an architectural reference, not a performance benchmark.

**Concepts studied:**
- Library API design for GPU kernels
- Template metaprogramming for kernel configuration
- How CUTLASS separates policy from mechanism
- What makes a GPU kernel genuinely reusable

---

## Benchmarking Methodology

### Timing

All GPU kernels are timed with **CUDA events**, not CPU-side timers.

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

CPU-side timers measure kernel launch overhead and synchronization delay. CUDA events measure **actual GPU execution time**. For performance engineering, only GPU time is meaningful.

### Protocol

- 5 warmup iterations before any timing begins
- 20 timed iterations minimum per configuration
- Results verified correct before any benchmark runs
- A kernel that produces wrong answers fast is not a fast kernel

### Reported Metrics

Every benchmark reports:

```
Matrix size        (M, N, K)
Runtime            mean / min / max / std deviation  (ms)
Achieved GFLOP/s
Theoretical peak   (GPU spec)
% of cuBLAS peak
```

### Benchmark Sizes

```
256  × 256
512  × 512
1024 × 1024
2048 × 2048
4096 × 4096
```

---

## Profiling

Every kernel is profiled with **Nsight Compute** before moving to the next stage. Intuition is not a profiling tool.

| Metric | What It Reveals |
|---|---|
| Achieved Occupancy | Are warps hiding latency effectively? |
| Global Memory Throughput | Are you bandwidth-limited? |
| Shared Memory Throughput | Are bank conflicts costing you? |
| L2 Hit Rate | Is your working set fitting in cache? |
| Arithmetic Intensity | Compute-bound or memory-bound? |
| FFMA Utilization | Are the ALUs actually busy? |
| Register Count per Thread | How much register pressure? |
| Warp Stall Reasons | Where exactly are threads waiting? |

Findings are recorded in [`profiling/nsight_notes.md`](profiling/nsight_notes.md) after each stage.

---

## Memory Layout

All matrices use contiguous **row-major** layout.

```
┌───────────────┐
│  1   2   3    │
│  4   5   6    │   →   [ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ]
└───────────────┘

Element access:  data[row * cols + col]
```

On the GPU, layout determines whether memory accesses **coalesce**. When threads in a warp access consecutive elements along a row, their requests merge into a single memory transaction. When they access down a column, each thread hits a separate cache line.

The performance difference is not theoretical. It is measured in Stage 1.

---

## Building

### Requirements

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 compatible compiler (GCC or Clang)
- NVIDIA GPU — Volta architecture or later recommended

### Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Run Correctness Tests

```bash
./correctness
```

### Run Benchmarks

```bash
./bench_kernels       # All CUDA kernel versions head-to-head
./bench_vs_cublas     # Gap analysis against cuBLAS
```

---

## Performance Engineering Philosophy

Three questions drive every decision in this project:

**1. Where is the bottleneck?**
Profile first. Never assume. Nsight Compute tells you what is actually happening on the hardware.

**2. What does the hardware want?**
Coalesced global memory access. No shared memory bank conflicts. Maximum warp occupancy. High FFMA utilization. These are not guidelines — they are the hardware contract.

**3. What does the number say?**
GFLOP/s as a percentage of cuBLAS peak is the only metric that matters. Every optimization either moves that number or it does not.

---

## License

MIT License