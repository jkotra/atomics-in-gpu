# GPU Floating-Point Atomics: SQLite vs. CAS Spin-Loop vs. Native FP32 Atomics

This repository contains executable benchmark code and OpenCL kernels accompanying the article **"Atomics in GPU"**.

It benchmarks a real-world aggregation problem across a **3,000,000 row fruit inventory dataset** (1M apples, 1M oranges, 1M bananas with individual weights) across three different execution engines:

1. **SQLite 3 (`GROUP BY name, SUM(weight)`)**: CPU row-oriented relational database engine running against `fruits.db`.
2. **GPU Emulated CAS Spin-Loop (`fruit_histogram_cas`)**: Traditional 32-bit floating-point addition via `atomic_cmpxchg` integer bit-reinterpretation directly in global VRAM.
3. **GPU Native Hardware FP32 Atomics (`fruit_histogram_native`)**: Direct hardware floating-point addition via the `cl_ext_float_atomics` extension (`atomic_fetch_add` on `atomic_float`).

---

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Mesa 26.3+ with Rusticl OpenCL support enabled (`rusticl` and `zink` / `vulkan-radeon`)

---

## Quickstart

```bash
# 1. Generate the synthetic fruit dataset (creates fruits.parquet and fruits.db)
uv run python generate_dataset.py

# 2. Run the benchmarks (SQLite vs. GPU CAS vs. GPU Native FP32)
RUSTICL_ENABLE=zink uv run python benchmark.py

# 3. Generate publication-quality 300-DPI comparison bar chart (log scale)
uv run python generate_plots.py
```

Generated plots:
- `plots/runtime_comparison.png`: Comparison bar chart with logarithmic execution time (ms) and speedup annotations (Lower is Better).

---

## Benchmark Results (3 Million Rows, 3 Bins)

Tested on **AMD Radeon RX 7600 (RDNA3, Navi 33)** with **Mesa Rusticl 26.3** via Zink over RADV:

| Engine | Median Runtime | Speedup vs. SQLite | Speedup vs. GPU CAS |
| :--- | :--- | :--- | :--- |
| **SQLite 3 `GROUP BY + SUM` (CPU)** | **2,567.66 ms** | 1.0x (Baseline) | — |
| **GPU Emulated CAS Loop (Kernel)** | **609.44 ms** | **4.2x faster** | 1.0x (Baseline) |
| **GPU Native FP32 Atomics (Total incl. Upload)** | **9.91 ms** | **259.2x faster** | **61.5x faster** |
| **GPU Native FP32 Atomics (Pure Kernel)** | **1.08 ms** | **2,377.6x faster** | **564.3x faster** |
