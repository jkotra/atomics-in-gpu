# GPU Floating-Point Atomics: CAS Spin-Loop vs. Native FP32 Atomics

This repository contains the executable benchmark code and OpenCL kernels accompanying the article **"Atomics in GPU"**.

It benchmarks two different strategies for floating-point accumulation on modern GPUs using **Mesa Rusticl**:

1. **Emulated CAS Spin-Loop (`histogram_cas`)**: Traditional 32-bit floating-point addition via `atomic_cmpxchg` integer bit-reinterpretation directly in global VRAM.
2. **Native Hardware FP32 Atomics (`histogram_native`)**: Direct hardware floating-point addition via the `cl_ext_float_atomics` extension (`atomic_fetch_add` on `atomic_float`).

---

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Mesa 26.3+ with Rusticl OpenCL support enabled (`rusticl` and `zink` / `vulkan-radeon`)

---

## Running the Benchmark

```bash
# 1. Run the benchmarks across contention scenarios (Single Bin, 64 Bins, 256 Bins, 1024 Bins)
RUSTICL_ENABLE=zink uv run python benchmark.py

# 2. Generate publication-quality 300-DPI bar chart (log scale)
uv run python generate_plots.py
```

Generated plot is saved to:
- `plots/runtime_comparison.png`: Side-by-side execution time in milliseconds (log scale for clear comparison across high and low contention).
