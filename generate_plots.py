#!/usr/bin/env python3
"""
Generate High-DPI Publication Quality Benchmark Plot:
SQLite GROUP BY vs. GPU Emulated CAS vs. GPU Native FP32 Atomics.
Saves runtime_comparison.png (Log scale on Y-axis for clear visibility)
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
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10,
        "legend.fontsize": 10.5,
        "figure.titlesize": 14,
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

    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    blog_img_dir = "/home/jkotra/Documents/Projects/jkotra.github.io/content/images/atomics-in-gpu"
    os.makedirs(blog_img_dir, exist_ok=True)

    sqlite_ms = data["sqlite_ms"]
    cas_ms = data["cas_kernel_ms"]
    native_ms = data["native_kernel_ms"]

    categories = [
        "SQLite 3\nGROUP BY + SUM\n(CPU)",
        "GPU Emulated\nCAS Spin-Loop",
        "GPU Native\nFP32 Atomics"
    ]
    runtimes = [sqlite_ms, cas_ms, native_ms]
    
    colors = [
        "#7f8c8d",  # Neutral Slate Gray for SQLite (CPU)
        "#e74c3c",  # Vibrant Coral Red for CAS Spin-Loop
        "#27ae60",  # Forest Emerald Green for Native FP32 Atomics
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(categories))
    width = 0.48

    bars = ax.bar(x, runtimes, width=width, color=colors, edgecolor="#2c3e50", linewidth=1.0)

    ax.set_ylabel("Execution Time (ms) [Log Scale, Lower is Better]")
    ax.set_title("Aggregation Runtime: SQLite vs. GPU CAS vs. Native FP32 Atomics\n(3 Million Fruit Inventory Rows into 3 Bins — Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_yscale("log")
    ax.set_ylim(0.1, max(runtimes) * 8.0)

    # Annotations on top of bars
    for bar, val in zip(bars, runtimes):
        h = bar.get_height()
        if val >= 100:
            txt = f"{val:,.1f} ms"
        elif val >= 1.0:
            txt = f"{val:.1f} ms"
        else:
            txt = f"{val:.2f} ms"
        
        # Calculate speedup relative to SQLite
        speedup = sqlite_ms / val
        if speedup >= 1.05:
            txt += f"\n({speedup:.1f}x faster)"
        elif speedup <= 0.95:
            txt += f"\n({1/speedup:.1f}x slower)"
        else:
            txt += "\n(Baseline)"

        ax.annotate(txt, xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    
    runtime_plot_path = os.path.join(plot_dir, "runtime_comparison.png")
    plt.savefig(runtime_plot_path)
    plt.close()
    print(f"Saved plot to {runtime_plot_path}")

    # Copy to blog assets
    shutil.copy(runtime_plot_path, os.path.join(blog_img_dir, "runtime_comparison.png"))
    print(f"Copied plot to blog directory: {blog_img_dir}")

if __name__ == "__main__":
    main()
