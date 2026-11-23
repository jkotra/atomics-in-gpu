#!/usr/bin/env python3
"""
Generate a 3-Million Row Fruit Dataset (1M Apple, 1M Orange, 1M Banana)
Saves to fruits.parquet and ingests into SQLite fruits.db.
"""

import os
import sqlite3
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def generate_data(n_per_fruit=1_000_000):
    total_rows = n_per_fruit * 3
    print(f"Generating synthetic fruit dataset: {total_rows:,} total rows ({n_per_fruit:,} per fruit)...")

    np.random.seed(42)

    # 1. Apples
    apple_types = np.array(["Fuji", "Gala", "Honeycrisp", "Granny Smith"])
    apple_type_col = np.random.choice(apple_types, size=n_per_fruit)
    apple_weights = np.random.normal(loc=180.0, scale=20.0, size=n_per_fruit).astype(np.float32)

    # 2. Oranges
    orange_types = np.array(["Navel", "Valencia", "Blood Orange", "Cara Cara"])
    orange_type_col = np.random.choice(orange_types, size=n_per_fruit)
    orange_weights = np.random.normal(loc=210.0, scale=25.0, size=n_per_fruit).astype(np.float32)

    # 3. Bananas
    banana_types = np.array(["Cavendish", "Plantain", "Red Banana", "Lady Finger"])
    banana_type_col = np.random.choice(banana_types, size=n_per_fruit)
    banana_weights = np.random.normal(loc=130.0, scale=15.0, size=n_per_fruit).astype(np.float32)

    # Concatenate
    names = np.concatenate([
        np.full(n_per_fruit, "apple"),
        np.full(n_per_fruit, "orange"),
        np.full(n_per_fruit, "banana")
    ])
    types = np.concatenate([apple_type_col, orange_type_col, banana_type_col])
    weights = np.concatenate([apple_weights, orange_weights, banana_weights])

    # Shuffle uniformly so rows are realistically interleaved
    perm = np.random.permutation(total_rows)
    names = names[perm]
    types = types[perm]
    weights = weights[perm]

    return names, types, weights

def save_parquet(names, types, weights, out_path):
    print(f"Writing Parquet file to {out_path}...")
    t0 = time.perf_counter()
    table = pa.Table.from_arrays([
        pa.array(names, type=pa.string()),
        pa.array(types, type=pa.string()),
        pa.array(weights, type=pa.float32()),
    ], names=["name", "type", "weight"])
    pq.write_table(table, out_path, compression="snappy")
    print(f"  Parquet file saved in {time.perf_counter() - t0:.2f}s ({os.path.getsize(out_path) / (1024**2):.1f} MB)")

def ingest_sqlite(names, types, weights, db_path):
    print(f"Ingesting into SQLite database at {db_path}...")
    t0 = time.perf_counter()
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA cache_size = 100000;")
    cur.execute("""
        CREATE TABLE fruits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            weight REAL NOT NULL
        );
    """)

    # Fast chunked insert
    batch_size = 100_000
    total = len(names)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        rows = list(zip(names[i:end], types[i:end], [float(w) for w in weights[i:end]]))
        cur.executemany("INSERT INTO fruits (name, type, weight) VALUES (?, ?, ?)", rows)
        conn.commit()

    conn.close()
    print(f"  SQLite ingested in {time.perf_counter() - t0:.2f}s ({os.path.getsize(db_path) / (1024**2):.1f} MB)")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(base_dir, "fruits.parquet")
    db_path = os.path.join(base_dir, "fruits.db")

    names, types, weights = generate_data(n_per_fruit=1_000_000)
    save_parquet(names, types, weights, parquet_path)
    ingest_sqlite(names, types, weights, db_path)

    print("\nDataset preparation complete!")

if __name__ == "__main__":
    main()
