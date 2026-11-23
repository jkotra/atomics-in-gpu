#!/usr/bin/env python3
"""
Fruit Inventory Benchmark: SQLite GROUP BY vs. GPU Emulated CAS vs. GPU Native FP32 Atomics
Processes 3,000,000 fruit records (1M apple, 1M orange, 1M banana) from fruits.parquet.
Computes total weight per fruit category across all three execution engines.
"""

import json
import os
import sqlite3
import time

# Ensure Rusticl uses Zink over RADV so cl_ext_float_atomics is enabled
os.environ.setdefault("RUSTICL_ENABLE", "zink")

import numpy as np
import pyarrow.parquet as pq
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

def benchmark_sqlite(db_path, iters=5):
    """Measures median execution time of SQLite SELECT name, SUM(weight) GROUP BY name."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    query = "SELECT name, SUM(weight) FROM fruits GROUP BY name ORDER BY name;"
    
    # Warmup
    cur.execute(query)
    sample_results = dict(cur.fetchall())

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        cur.execute(query)
        _ = cur.fetchall()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    conn.close()
    return float(np.median(times)), sample_results

def run_benchmarks():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(base_dir, "fruits.parquet")
    db_path = os.path.join(base_dir, "fruits.db")

    if not os.path.exists(parquet_path) or not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Dataset files missing. Run generate_dataset.py first to create {parquet_path} and {db_path}."
        )

    print("=" * 88)
    print("Fruit Inventory Aggregation Benchmark: SQLite vs. GPU CAS vs. GPU Native FP32")
    print("=" * 88)

    # 1. Read Parquet dataset
    print(f"\n1. Reading Parquet dataset: {parquet_path}...")
    t0 = time.perf_counter()
    table = pq.read_table(parquet_path)
    parquet_read_ms = (time.perf_counter() - t0) * 1000.0

    raw_names = table["name"].to_numpy()
    weights = table["weight"].to_numpy().astype(np.float32)
    N = len(weights)

    # Map categories to integer bins: apple -> 0, orange -> 1, banana -> 2
    category_map = {"apple": 0, "orange": 1, "banana": 2}
    categories = np.empty(N, dtype=np.int32)
    for name, cid in category_map.items():
        categories[raw_names == name] = cid

    print(f"   Loaded {N:,} rows in {parquet_read_ms:.2f} ms")
    print(f"   Unique categories: {list(category_map.keys())}")

    # 2. Benchmark SQLite
    print(f"\n2. Benchmarking SQLite GROUP BY on {db_path}...")
    sqlite_ms, sqlite_sums = benchmark_sqlite(db_path, iters=5)
    print(f"   SQLite Median Runtime: {sqlite_ms:.2f} ms")
    for name, s in sqlite_sums.items():
        print(f"     - {name.capitalize():<8}: {s:,.2f} grams")

    # 3. Benchmark GPU Kernels
    print("\n3. Initializing Mesa Rusticl OpenCL...")
    platform, device, ctx, queue = get_opencl_context()
    print(f"   Platform: {platform.name}")
    print(f"   Device:   {device.name}")

    kernel_path = os.path.join(base_dir, "kernels.cl")
    with open(kernel_path, "r") as f:
        src = f.read()

    prg = cl.Program(ctx, src).build(options=["-cl-std=CL3.0"])
    k_cas = cl.Kernel(prg, "fruit_histogram_cas")
    k_native = cl.Kernel(prg, "fruit_histogram_native")

    mf = cl.mem_flags
    num_bins = len(category_map)

    # Measure Host-to-Device Memory Transfer
    t0 = time.perf_counter()
    d_categories = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=categories)
    d_weights = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights)
    queue.finish()
    h2d_ms = (time.perf_counter() - t0) * 1000.0
    print(f"   Host-to-Device Buffer Upload ({N * 8 / (1024**2):.1f} MB): {h2d_ms:.2f} ms")

    # Output buffers
    d_cas_out = cl.Buffer(ctx, mf.READ_WRITE, size=num_bins * 4)
    d_native_out = cl.Buffer(ctx, mf.READ_WRITE, size=num_bins * 4)

    gsize = (N,)

    # -------------------------------------------------------------------------
    # Profile CAS Spin-Loop Kernel
    # -------------------------------------------------------------------------
    print("\n4. Running GPU Emulated CAS Spin-Loop (atomic_cmpxchg)...")
    k_cas.set_arg(0, d_cas_out)
    k_cas.set_arg(1, d_categories)
    k_cas.set_arg(2, d_weights)
    k_cas.set_arg(3, np.int32(N))

    # Warmup
    cl.enqueue_fill_buffer(queue, d_cas_out, np.int32(0), 0, d_cas_out.size).wait()
    evt = cl.enqueue_nd_range_kernel(queue, k_cas, gsize, None)
    evt.wait()

    cas_times = []
    for _ in range(5):
        cl.enqueue_fill_buffer(queue, d_cas_out, np.int32(0), 0, d_cas_out.size).wait()
        evt = cl.enqueue_nd_range_kernel(queue, k_cas, gsize, None)
        evt.wait()
        cas_times.append((evt.profile.end - evt.profile.start) / 1e6)

    cas_kernel_ms = float(np.median(cas_times))

    # Read back CAS results
    h_cas_out = np.empty(num_bins, dtype=np.int32)
    cl.enqueue_copy(queue, h_cas_out, d_cas_out).wait()
    cas_sums = [float(np.frombuffer(h_cas_out[i:i+1], dtype=np.float32)[0]) for i in range(num_bins)]
    print(f"   CAS Kernel Median Runtime: {cas_kernel_ms:.2f} ms")

    # -------------------------------------------------------------------------
    # Profile Native FP32 Atomics Kernel
    # -------------------------------------------------------------------------
    print("\n5. Running GPU Native Hardware FP32 Atomics (cl_ext_float_atomics)...")
    k_native.set_arg(0, d_native_out)
    k_native.set_arg(1, d_categories)
    k_native.set_arg(2, d_weights)
    k_native.set_arg(3, np.int32(N))

    # Warmup
    cl.enqueue_fill_buffer(queue, d_native_out, np.int32(0), 0, d_native_out.size).wait()
    evt = cl.enqueue_nd_range_kernel(queue, k_native, gsize, None)
    evt.wait()

    native_times = []
    for _ in range(5):
        cl.enqueue_fill_buffer(queue, d_native_out, np.int32(0), 0, d_native_out.size).wait()
        evt = cl.enqueue_nd_range_kernel(queue, k_native, gsize, None)
        evt.wait()
        native_times.append((evt.profile.end - evt.profile.start) / 1e6)

    native_kernel_ms = float(np.median(native_times))

    # Read back Native results
    h_native_out = np.empty(num_bins, dtype=np.float32)
    cl.enqueue_copy(queue, h_native_out, d_native_out).wait()
    native_sums = [float(h_native_out[i]) for i in range(num_bins)]
    print(f"   Native FP32 Kernel Median Runtime: {native_kernel_ms:.2f} ms")

    # Summary Table
    print("\n" + "=" * 88)
    print("BENCHMARK RESULTS SUMMARY (3 Million Rows, 3 Bins)")
    print("=" * 88)
    print(f"{'Engine':<40} | {'Median Runtime (ms)':<22} | {'Relative to SQLite':<20}")
    print("-" * 88)
    print(f"{'SQLite GROUP BY + SUM (CPU)':<40} | {sqlite_ms:>18.2f} ms | {'1.0x (baseline)':<20}")
    print(f"{'GPU Emulated CAS Loop (Kernel)':<40} | {cas_kernel_ms:>18.2f} ms | {sqlite_ms / cas_kernel_ms:>18.1f}x faster")
    print(f"{'GPU Native Hardware FP32 (Kernel)':<40} | {native_kernel_ms:>18.2f} ms | {sqlite_ms / native_kernel_ms:>18.1f}x faster")
    print(f"{'GPU Native Hardware FP32 (incl. H2D)':<40} | {native_kernel_ms + h2d_ms:>18.2f} ms | {sqlite_ms / (native_kernel_ms + h2d_ms):>18.1f}x faster")
    print("-" * 88)
    print(f"Speedup of Native FP32 over CAS Loop: {cas_kernel_ms / native_kernel_ms:.1f}x faster")
    print("=" * 88)

    # Verification vs SQLite
    print("\nNumerical Accuracy Verification:")
    fruits = ["apple", "orange", "banana"]
    for i, fruit in enumerate(fruits):
        sql_val = sqlite_sums[fruit]
        cas_val = cas_sums[i]
        nat_val = native_sums[i]
        diff_cas = abs(cas_val - sql_val) / sql_val * 100.0
        diff_nat = abs(nat_val - sql_val) / sql_val * 100.0
        print(f"  {fruit.capitalize():<8} -> SQLite: {sql_val:,.1f}g | CAS: {cas_val:,.1f}g (diff {diff_cas:.4f}%) | Native: {nat_val:,.1f}g (diff {diff_nat:.4f}%)")

    # Save to JSON
    output_data = {
        "device": device.name,
        "platform": platform.name,
        "num_rows": N,
        "num_bins": num_bins,
        "h2d_transfer_ms": h2d_ms,
        "sqlite_ms": sqlite_ms,
        "cas_kernel_ms": cas_kernel_ms,
        "native_kernel_ms": native_kernel_ms,
        "native_total_ms": native_kernel_ms + h2d_ms,
        "cas_speedup_over_sqlite": sqlite_ms / cas_kernel_ms,
        "native_speedup_over_sqlite": sqlite_ms / native_kernel_ms,
        "native_speedup_over_cas": cas_kernel_ms / native_kernel_ms,
        "verification": {
            fruit: {
                "sqlite_sum": sqlite_sums[fruit],
                "cas_sum": cas_sums[i],
                "native_sum": native_sums[i]
            }
            for i, fruit in enumerate(fruits)
        }
    }

    out_json = os.path.join(base_dir, "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved detailed benchmark results to {out_json}")

if __name__ == "__main__":
    run_benchmarks()
