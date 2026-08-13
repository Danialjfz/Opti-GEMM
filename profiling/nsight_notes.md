# Nsight Compute Profiling Notes — Opti-GEMM

Workflows for profiling the GEMM kernels on three environments: local NVIDIA GPU,
Google Colab, and Kaggle. The ready-made Colab notebook lives in
[`notebooks/opti_gemm_colab.ipynb`](../notebooks/opti_gemm_colab.ipynb).

---

## 1. The profile-on-cloud / analyze-locally loop

Cloud GPU time is for *collecting*; analysis is nicer in the desktop GUI.

```bash
# On the GPU machine (local, Colab, Kaggle):
ncu --set full --force-overwrite     --export opti_gemm_full     --kernel-name regex:"(naive|tiled)_gemm"     --launch-skip 20 --launch-count 2     ./build/bench_kernels
```

Then download `opti_gemm_full.ncu-rep` and open it in Nsight Compute locally
(free download from NVIDIA; no GPU required to *view* reports). The build already
compiles with `-lineinfo`, so the Source page correlates metrics with CUDA source lines.

---

## 2. Measurement caveats (learned the hard way)

- **Never quote benchmark rows from a profiled run.** `ncu` replays each profiled
  launch several times (the `8 passes` progress lines). If a profiled launch falls
  inside a timed benchmark iteration, that row's mean/stddev are meaningless.
  Collect timings unprofiled; collect metrics under `ncu`; never both at once.
- **`ncu` locks clocks to base by default** (`--clock-control base` → 585 MHz on a
  T4, vs ~1590 MHz boost). Durations in reports are ~2× slower than production.
  Use `--clock-control none` when wall-clock representativeness matters more than
  run-to-run reproducibility.
- **"Compute (SM) Throughput" is an aggregate**, not the FP32 pipe. A kernel doing
  2 global loads per FMA (naive GEMM) can show ~80% "compute" while the FMA pipe
  idles at ~30%. Check `--section ComputeWorkloadAnalysis` for per-pipe truth.
- **Problem size changes the bottleneck.** At 512³ the working set is L2-resident
  (DRAM ≈ 1% busy) and the limiter is the L1/TEX pipe; at 4096³ DRAM traffic finally
  matters. Profile the size you intend to draw conclusions about.

## 3. Environment specifics

### Local NVIDIA GPU
`ncu` ships with the CUDA toolkit (`/usr/local/cuda/bin/ncu`). Profiling needs
counter permission; on Linux, if you get `ERR_NVGPUCTRPERM`, either run once with
`sudo` or set the driver module option:

```bash
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
# reload the driver (reboot is simplest)
```

### Google Colab (free T4 / paid L4, A100)
- CUDA toolkit + `ncu` + `nsys` are preinstalled.
- Runtime → Change runtime type → GPU.
- If counters are blocked (`ERR_NVGPUCTRPERM`), factory-reset the runtime and retry —
  permission depends on the instance you land on.

### Kaggle (free P100 — Pascal!)
- ~30 h/week GPU quota. The P100 reproduces the README's Pascal numbers, where
  Tiled-SMEM beats Naive ~5× because Pascal doesn't cache global loads in L1.
- Same notebook cells work; `ncu` may need an apt install.

---

## 4. Command cheat sheet

| Goal | Command |
|---|---|
| Compute- or memory-bound? | `ncu --section SpeedOfLight ./build/bench_kernels` |
| Memory traffic breakdown | `ncu --section MemoryWorkloadAnalysis ...` |
| Why are warps stalled? | `ncu --section WarpStateStats ...` |
| Occupancy limiters | `ncu --section Occupancy ...` |
| Everything (slow, uses replay) | `ncu --set full --export report ...` |
| Only some kernels/launches | `--kernel-name regex:"tiled_gemm" --launch-skip 20 --launch-count 3` |
| Single metrics | `--metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum` |
| Timeline across kernels | `nsys profile -o timeline ./build/bench_kernels` |

## 5. Metrics that matter for GEMM

| Metric | What it tells you |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Compute (SOL) utilization — how close to FMA peak |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | Memory (SOL) utilization — DRAM/L2 pressure |
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | Warps waiting on global memory — the Naive kernel's disease |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` | Shared-memory bank conflicts — should be ~0 for the padded Tiled kernel |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy — too low means not enough parallelism to hide latency |
| `launch__registers_per_thread` | Register pressure — the key limiter once Stage 3 register blocking lands |

## 6. Reading Speed-of-Light for this project

- **Naive GEMM:** memory SOL ≫ compute SOL. Each thread re-reads a full row of A
  and a full column of B from global memory — 2·K global loads per output element.
  On Pascal (no L1 caching of globals) this is catastrophic; on Turing the unified
  L1 partially rescues it, which is exactly the architecture difference documented
  in the README benchmark tables.
- **Tiled-SMEM:** memory SOL drops, compute SOL rises. 16×16 tiles turn 16 global
  loads per operand into 1, at the cost of `__syncthreads()` barriers.
- **Endgame (Stages 3–4):** push compute SOL toward 60–80% by increasing arithmetic
  intensity per thread (register blocking) and per warp (warp tiling). cuBLAS-class
  kernels sit compute-bound at ~90% of SOL — that gap is the project's final story.

## 7. What to capture per kernel stage

For each new kernel, record in `docs/kernel_analysis.md`:

1. Speed-of-Light screenshot (compute vs memory %)
2. Top-3 warp stall reasons
3. Achieved occupancy + its limiter (registers / shared mem / block size)
4. GFLOP/s vs theoretical peak at 4096³
5. One sentence: *what changed, and what the hardware did about it*
