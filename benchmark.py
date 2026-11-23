#!/usr/bin/env python3
"""
GPU Atomics Benchmark: Emulated CAS Spin-Loop vs. Native Hardware FP32 Atomics
Measures runtime, throughput, and speedup on Mesa Rusticl with cl_ext_float_atomics.
"""

import json
import os
import numpy as np
import pyopencl as cl

def get_opencl_context():
    """Finds and initializes Mesa Rusticl device (preferring Zink or AMD GPU)."""
    platforms = cl.get_platforms()
    rusticl_plat = None
    for p in platforms:
        if "rusticl" in p.name.lower():
            rusticl_plat = p
            break
    if rusticl_plat is None:
        rusticl_plat = platforms[0]

    devices = rusticl_plat.get_devices()
    selected_dev = devices[0]
    for d in devices:
        if "zink" in d.name.lower() or "radeon" in d.name.lower():
            selected_dev = d
            break

    ctx = cl.Context([selected_dev])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    return rusticl_plat, selected_dev, ctx, queue

def profile_kernel(queue, kernel, args, d_out, gsize, iters=3):
    """Measures median execution time of an OpenCL kernel in milliseconds."""
    for idx, arg in enumerate(args):
        kernel.set_arg(idx, arg)
    
    # Zero buffer before warmup
    cl.enqueue_fill_buffer(queue, d_out, np.int32(0), 0, d_out.size).wait()
    evt = cl.enqueue_nd_range_kernel(queue, kernel, gsize, None)
    evt.wait()

    times = []
    for _ in range(iters):
        cl.enqueue_fill_buffer(queue, d_out, np.int32(0), 0, d_out.size).wait()
        evt = cl.enqueue_nd_range_kernel(queue, kernel, gsize, None)
        evt.wait()
        times.append((evt.profile.end - evt.profile.start) / 1e6)
    
    return float(np.median(times))

def run_benchmarks():
    platform, device, ctx, queue = get_opencl_context()
    print("=" * 88)
    print(f"GPU FP32 Atomics Benchmark: {device.name}")
    print(f"Platform: {platform.name} (Testing CAS vs Native cl_ext_float_atomics)")
    print("=" * 88)

    # Load kernel source
    kernel_path = os.path.join(os.path.dirname(__file__), "kernels.cl")
    with open(kernel_path, "r") as f:
        src = f.read()

    prg = cl.Program(ctx, src).build(options=["-cl-std=CL3.0"])
    k_cas = cl.Kernel(prg, "histogram_cas")
    k_native = cl.Kernel(prg, "histogram_native")

    mf = cl.mem_flags
    N = 10_000_000
    h_in = np.random.uniform(0.1, 1.0, size=N).astype(np.float32)
    d_in = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_in)

    scenarios = [
        {"bins": 1, "label": "Single Bin (Extreme Contention)"},
        {"bins": 64, "label": "64 Bins (Realistic Simulation Pattern)"},
        {"bins": 256, "label": "256 Bins (Moderate Contention)"},
        {"bins": 1024, "label": "1024 Bins (Low Contention)"},
    ]

    results = []

    print(f"\nProcessing {N:,} elements ({N * 4 / (1024**2):.1f} MB float32)...")
    print("-" * 90)
    print(f"{'Contention Level':<38} | {'CAS Loop':<12} | {'Native FP32':<14} | {'Speedup':<10}")
    print("-" * 90)

    for sc in scenarios:
        num_bins = sc["bins"]
        label = sc["label"]

        # Buffers for CAS
        h_cas = np.zeros(num_bins, dtype=np.int32)
        d_cas = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_cas)

        # Buffers for Native
        h_native = np.zeros(num_bins, dtype=np.float32)
        d_native = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_native)

        # CAS (1 iteration for 1-bin to prevent long stall)
        cas_iters = 1 if num_bins == 1 else 3
        t_cas = profile_kernel(queue, k_cas, [d_cas, d_in, np.int32(num_bins), np.int32(N)], d_cas, (N,), iters=cas_iters)

        # Native FP32
        t_native = profile_kernel(queue, k_native, [d_native, d_in, np.int32(num_bins), np.int32(N)], d_native, (N,), iters=3)

        speedup = t_cas / t_native if t_native > 0 else 0.0

        # Verification: compare sums
        cl.enqueue_copy(queue, h_cas, d_cas).wait()
        cl.enqueue_copy(queue, h_native, d_native).wait()
        sum_cas = float(np.sum(h_cas.view(np.float32), dtype=np.float64))
        sum_native = float(np.sum(h_native, dtype=np.float64))
        assert np.isclose(sum_cas, sum_native, rtol=1e-2), f"Mismatch: {sum_cas} vs {sum_native}"

        row = {
            "bins": num_bins,
            "label": label,
            "cas_ms": t_cas,
            "native_ms": t_native,
            "speedup": speedup,
            "cas_mops": (N / (t_cas / 1000.0)) / 1e6,
            "native_mops": (N / (t_native / 1000.0)) / 1e6,
        }
        results.append(row)

        print(f"{label:<38} | {t_cas:>9.2f} ms | {t_native:>11.2f} ms | {speedup:>9.1f}x")

    # Save to JSON
    out_json = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump({"device": device.name, "elements": N, "results": results}, f, indent=2)
    print("\nBenchmark results saved to:", out_json)
    return results

if __name__ == "__main__":
    run_benchmarks()
