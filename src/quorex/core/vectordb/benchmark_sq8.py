"""
benchmark_sq8.py
-----------------
Compares standard float32 HNSW vs SQ8-quantized HNSW on:
  - RAM usage      (tracemalloc peak)
  - Insert time    (N vectors)
  - Search latency (1 000 queries)
  - Recall@10      (vs brute-force ground truth)

Run:
    python3 src/quorex/core/vectordb/benchmark_sq8.py
"""

from __future__ import annotations

import os
import sys
import shutil
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from quorex.core.vectordb.hnsw import HNSWIndex
from quorex.core.vectordb.quantizer import SQ8Quantizer

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

N_VECTORS  = 10_000
DIM        = 128
N_QUERIES  = 1_000
TOP_K      = 10
SEED       = 42
M          = 16
EF_CONSTR  = 200
EF_SEARCH  = 50

rng = np.random.default_rng(SEED)


def _gen_vectors(n: int, dim: int) -> np.ndarray:
    """Unit-normalized float32 vectors — realistic embedding distribution."""
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _brute_force_top_k(
    queries: np.ndarray, corpus: np.ndarray, k: int
) -> list[set[int]]:
    """Exact cosine nearest neighbors for ground truth (queries @ corpus.T)."""
    sims = queries @ corpus.T                          # (n_q, n_corpus)
    gt   = []
    for row in sims:
        top = set(np.argpartition(row, -k)[-k:])
        gt.append(top)
    return gt


# -----------------------------------------------------------------------
# Benchmark helpers
# -----------------------------------------------------------------------

def _build_index(vectors: np.ndarray, quantizer=None) -> HNSWIndex:
    idx = HNSWIndex(
        dim=DIM, M=M, ef_construction=EF_CONSTR,
        ef_search=EF_SEARCH, seed=SEED, quantizer=quantizer
    )
    for i, v in enumerate(vectors):
        idx.insert(i, v)
    return idx


def bench_insert(vectors: np.ndarray, quantizer=None) -> tuple[float, int]:
    """Returns (elapsed_seconds, peak_bytes)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    idx = _build_index(vectors, quantizer)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del idx
    return elapsed, peak


def bench_search(
    idx: HNSWIndex, queries: np.ndarray, n_queries: int
) -> float:
    """Returns average search latency in milliseconds."""
    t0 = time.perf_counter()
    for q in queries[:n_queries]:
        idx.search(q, top_k=TOP_K)
    elapsed = time.perf_counter() - t0
    return elapsed / n_queries * 1000.0


def recall_at_k(
    idx: HNSWIndex, queries: np.ndarray, gt: list[set[int]], k: int
) -> float:
    hits = 0
    total = 0
    for q, true_set in zip(queries, gt):
        result_ids = {nid for nid, _ in idx.search(q, top_k=k)}
        hits  += len(result_ids & true_set)
        total += len(true_set)
    return hits / total if total else 0.0


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    print(f"Generating {N_VECTORS:,} vectors (dim={DIM})...")
    corpus  = _gen_vectors(N_VECTORS, DIM)
    queries = _gen_vectors(N_QUERIES, DIM)

    print(f"Computing brute-force ground truth (Recall@{TOP_K})...")
    gt = _brute_force_top_k(queries, corpus, TOP_K)

    # ---- Fit quantizer on a sample of the corpus ----
    print("Fitting SQ8 quantizer...")
    quantizer = SQ8Quantizer()
    quantizer.fit(list(corpus[:2_000]))

    # ---- Insert benchmarks ----
    print(f"\nInserting {N_VECTORS:,} vectors into standard index...")
    t_std, ram_std   = bench_insert(corpus)

    print(f"Inserting {N_VECTORS:,} vectors into SQ8 index...")
    t_sq8, ram_sq8   = bench_insert(corpus, quantizer=quantizer)

    # ---- Build final indexes for search / recall ----
    print("Building final indexes for search benchmark...")
    idx_std  = _build_index(corpus)
    idx_sq8  = _build_index(corpus, quantizer=quantizer)

    # ---- Search latency ----
    print(f"Measuring search latency ({N_QUERIES:,} queries)...")
    lat_std  = bench_search(idx_std,  queries, N_QUERIES)
    lat_sq8  = bench_search(idx_sq8,  queries, N_QUERIES)

    # ---- Recall@K ----
    print(f"Measuring Recall@{TOP_K}...")
    rec_std  = recall_at_k(idx_std,  queries, gt, TOP_K)
    rec_sq8  = recall_at_k(idx_sq8,  queries, gt, TOP_K)

    # ---- Node RAM estimate (more accurate than tracemalloc for numpy) ----
    # Each float32 node: dim * 4 bytes.  Each uint8 node: dim * 1 byte.
    node_ram_std = N_VECTORS * DIM * 4
    node_ram_sq8 = N_VECTORS * DIM * 1

    # ---- Print table ----
    W = 54

    def pct(a, b):
        if b == 0:
            return "n/a"
        d = (a - b) / b * 100
        return f"{d:+.1f}%"

    def fmt_bytes(n):
        if n >= 1_048_576:
            return f"{n / 1_048_576:.1f} MB"
        return f"{n / 1024:.1f} KB"

    print()
    print("─" * W)
    print(f"  {'Metric':<24} {'Standard':>10} {'SQ8':>10} {'Delta':>8}")
    print("─" * W)
    print(f"  {'Vector RAM (exact)':<24} {fmt_bytes(node_ram_std):>10} {fmt_bytes(node_ram_sq8):>10} {pct(node_ram_sq8, node_ram_std):>8}")
    print(f"  {'Peak alloc (build)':<24} {fmt_bytes(ram_std):>10} {fmt_bytes(ram_sq8):>10} {pct(ram_sq8, ram_std):>8}")
    print(f"  {'Insert time':>24s} {t_std:>9.2f}s {t_sq8:>9.2f}s {pct(t_sq8, t_std):>8}")
    print(f"  {'Search latency':>24s} {lat_std:>8.3f}ms {lat_sq8:>8.3f}ms {pct(lat_sq8, lat_std):>8}")
    print(f"  {f'Recall@{TOP_K}':<24} {rec_std:>9.1%} {rec_sq8:>9.1%} {pct(rec_sq8, rec_std):>8}")
    print("─" * W)
    print(f"  Compression ratio : {quantizer.compression_ratio}x  (float32 → uint8)")
    print()


if __name__ == "__main__":
    main()
