#!/usr/bin/env python3
"""
Generate High-DPI Publication Quality Benchmark Plot: CAS vs. Native FP32 Atomics.
Saves runtime_comparison.png (using a logarithmic scale on Y-axis for clear visibility)
to atomics-in-gpu/plots/ and copies to the blog assets directory.
"""

import json
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

def set_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10.5,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

def main():
    set_plot_style()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "benchmark_results.json")
    
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data["results"]

    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    blog_img_dir = "/home/jkotra/Documents/Projects/jkotra.github.io/content/images/atomics-in-gpu"
    blog_static_dir = "/home/jkotra/Documents/Projects/jkotra.github.io/static/images/atomics-in-gpu"
    os.makedirs(blog_img_dir, exist_ok=True)
    os.makedirs(blog_static_dir, exist_ok=True)

    scenarios = [r["label"].split(" (")[0] for r in results]
    x = np.arange(len(scenarios))
    width = 0.35

    cas_times = [r["cas_ms"] for r in results]
    nat_times = [r["native_ms"] for r in results]

    c_cas = "#e74c3c"   # Vibrant Red
    c_nat = "#2980b9"   # Modern Deep Blue

    # -------------------------------------------------------------------------
    # Runtime Comparison (Log Scale on Y-Axis)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    rects1 = ax.bar(x - width/2, cas_times, width, label="Emulated CAS Loop (atomic_cmpxchg)", 
                    color=c_cas, edgecolor="#2c3e50", linewidth=0.8)
    rects2 = ax.bar(x + width/2, nat_times, width, label="Native Hardware Atomics (cl_ext_float_atomics)", 
                    color=c_nat, edgecolor="#2c3e50", linewidth=0.8)

    ax.set_ylabel("Execution Time (ms) [Log Scale, Lower is Better]")
    ax.set_title("Kernel Runtime: Emulated CAS Loop vs. Native FP32 Atomics (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_yscale("log")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    # Annotate CAS bars
    for bar in rects1:
        h = bar.get_height()
        txt = f"{h:,.1f} ms" if h >= 100 else (f"{h:.1f} ms" if h >= 1.0 else f"{h:.2f} ms")
        ax.annotate(txt, xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Annotate Native bars
    for bar in rects2:
        h = bar.get_height()
        txt = f"{h:.2f} ms" if h < 1 else f"{h:.1f} ms"
        ax.annotate(txt, xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1a5276")

    ax.set_ylim(0.1, 15000)
    plt.tight_layout()
    
    runtime_plot_path = os.path.join(plot_dir, "runtime_comparison.png")
    plt.savefig(runtime_plot_path)
    plt.close()

    # Copy to blog assets
    shutil.copy2(runtime_plot_path, os.path.join(blog_img_dir, "runtime_comparison.png"))
    shutil.copy2(runtime_plot_path, os.path.join(blog_static_dir, "runtime_comparison.png"))

    print("\nUpdated High-DPI runtime plot generated with log scale Y-axis:")
    print(" -", runtime_plot_path)
    print("Copied to blog assets in:", blog_img_dir)

if __name__ == "__main__":
    main()
