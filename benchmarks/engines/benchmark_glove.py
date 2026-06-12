"""
benchmarks/benchmark_glove.py
──────────────────────────────
Benchmark multi-engine sur GloVe 6B (données texte réelles).

Remplace les vecteurs gaussiens par de vrais embeddings de mots.
GloVe a une structure sémantique forte (clusters, corrélations) —
bien plus représentatif des embeddings en production que les gaussiens.

Paliers : 10K → 25K → 50K → 100K vecteurs

Usage :
  python -m benchmarks.benchmark_glove \
      --glove /path/to/glove.6B.100d.txt \
      --max-n 100000 --steps 4 \
      --csv results/benchmark_glove.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.engines.engine_quorex_baseline import QuorexBaselineEngine
from benchmarks.engines.engine_quorex         import QuorexEngine, QuorexOptimizedEngine
from benchmarks.engines.engine_hnswlib         import HnswlibEngine
from benchmarks.engines.engine_faiss_flat      import FaissHNSWFlatEngine
from benchmarks.engines.engine_faiss_sq8       import FaissIVFSQ8Engine

SEED = 42

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BenchmarkPoint:
    engine:         str
    n_vectors:      int
    ram_mb:         float
    p50_ms:         float
    p95_ms:         float
    p99_ms:         float
    recall_at_10:   float
    throughput_rps: float
    implementation: str

@dataclass
class EngineResult:
    name:   str
    points: list[BenchmarkPoint] = field(default_factory=list)

# ── Chargement GloVe ──────────────────────────────────────────────────────────

def load_glove(path: str, max_n: int) -> np.ndarray:
    """
    Charge les max_n premiers vecteurs GloVe.
    Normalise en float32 unitaire (cosine similarity).
    """
    print(f"Chargement GloVe ({max_n:,} vecteurs max) depuis {path}...")
    vecs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            parts = line.strip().split()
            # premier token = le mot, le reste = le vecteur
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                if len(vec) == 0:
                    continue
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vecs.append(vec / norm)
            except ValueError:
                continue
            if (i + 1) % 10_000 == 0:
                print(f"  {i+1:,} vecteurs chargés...")

    matrix = np.stack(vecs, axis=0)
    print(f"  ✓ {len(matrix):,} vecteurs chargés, dim={matrix.shape[1]}")
    return matrix

# ── Dataset ───────────────────────────────────────────────────────────────────

def make_queries(vecs: np.ndarray, n_queries: int = 200) -> np.ndarray:
    """Tire n_queries vecteurs du corpus comme requêtes."""
    rng = np.random.default_rng(SEED + 1)
    n_q = min(n_queries, len(vecs))
    idx = rng.choice(len(vecs), size=n_q, replace=False)
    return vecs[idx].copy()

def ground_truth(vecs: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    """k-NN exact par force brute (cosine similarity)."""
    # En batch pour éviter les OOM sur grandes matrices
    batch = 50
    gt = []
    for i in range(0, len(queries), batch):
        q_batch = queries[i:i+batch]
        scores  = q_batch @ vecs.T
        gt.append(np.argsort(-scores, axis=1)[:, :k])
    return np.vstack(gt)

def recall_at_k(retrieved: list[list[int]], gt: np.ndarray, k: int = 10) -> float:
    hits = total = 0
    for ret, g in zip(retrieved, gt):
        hits  += len(set(ret[:k]) & set(g[:k].tolist()))
        total += k
    return hits / total if total else 0.0

# ── Benchmark d'un engine ─────────────────────────────────────────────────────

def bench_engine(engine, vecs, queries, gt, n, k=10, impl="Python") -> BenchmarkPoint:
    gc.collect()
    engine.build(vecs[:n])
    ram = engine.ram_mb()

    lats, retrieved = [], []
    for q in queries:
        t0  = time.perf_counter()
        ids = engine.search(q, top_k=k)
        lats.append((time.perf_counter() - t0) * 1000)
        retrieved.append(ids)

    engine.destroy()

    a = np.array(lats)
    return BenchmarkPoint(
        engine=engine.name,
        n_vectors=n,
        ram_mb=ram,
        p50_ms=float(np.percentile(a, 50)),
        p95_ms=float(np.percentile(a, 95)),
        p99_ms=float(np.percentile(a, 99)),
        recall_at_10=recall_at_k(retrieved, gt, k),
        throughput_rps=1000.0 / float(np.mean(a)),
        implementation=impl,
    )

# ── Runner ────────────────────────────────────────────────────────────────────

def run_suite(
    glove_path: str,
    max_n: int = 100_000,
    steps: int = 4,
    k: int = 10,
    M: int = 16,
    ef: int = 50,
    n_queries: int = 200,
) -> list[EngineResult]:

    # Charge tout GloVe une seule fois
    all_vecs = load_glove(glove_path, max_n)
    dim      = all_vecs.shape[1]
    actual_n = len(all_vecs)

    # Paliers géométriques
    ns = np.geomspace(
        min(10_000, actual_n),
        actual_n,
        steps
    ).astype(int).tolist()
    # déduplique
    ns = sorted(set(ns))

    engines_cfg = [
        (QuorexBaselineEngine(M=M, ef_construction=200, ef_search=ef),   "Python"),
        (QuorexEngine(M=M, ef_construction=200, ef_search=ef),            "Python"),
        (QuorexOptimizedEngine(M=M, ef_construction=200),                 "Python"),
        (HnswlibEngine(M=M, ef_construction=200, ef_search=ef),           "C++"),
        (FaissHNSWFlatEngine(M=M, ef_search=ef),                          "C++"),
        (FaissIVFSQ8Engine(
            nlist=max(4, min(256, actual_n // 40)),
            nprobe=10
        ), "C++"),
    ]

    results = {e.name: EngineResult(name=e.name) for e, _ in engines_cfg}

    print(f"\n{'='*76}")
    print(
        f"  BENCHMARK GLOVE  |  dim={dim}  max_n={actual_n:,}  "
        f"k={k}  M={M}  ef={ef}"
    )
    print(f"{'='*76}\n")
    print("Engines :")
    for e, impl in engines_cfg:
        print(f"  [{impl:6}] {e.name}")

    for n in ns:
        vecs    = all_vecs[:n]
        queries = make_queries(vecs, n_queries)
        print(f"\n▶ Ground truth brute-force N={n:,}...")
        gt = ground_truth(vecs, queries, k)

        print(f"\n── N = {n:>7,} vectors  ({len(queries)} queries) ──────────────────")

        for engine, impl in engines_cfg:
            print(f"  {engine.name}...")
            try:
                pt = bench_engine(engine, vecs, queries, gt, n, k, impl)
                results[engine.name].points.append(pt)
                print(
                    f"    [{impl:6}] RAM={pt.ram_mb:.1f}MB  "
                    f"p50={pt.p50_ms:.3f}ms  "
                    f"Recall@{k}={pt.recall_at_10:.3%}"
                )
            except Exception as ex:
                print(f"    ✗ ERREUR : {ex}")

    return list(results.values())

# ── Rapport ───────────────────────────────────────────────────────────────────

def print_report(results: list[EngineResult], k: int = 10) -> None:
    print(f"\n{'='*76}")
    print("  RAPPORT FINAL — BENCHMARK GLOVE")
    print(f"{'='*76}")

    hdr = (
        f"{'N':>8} | {'Impl':>6} | {'RAM (MB)':>10} | "
        f"{'p50 (ms)':>10} | {'p99 (ms)':>10} | "
        f"{f'Recall@{k}':>10} | {'RPS':>8}"
    )
    sep = "-" * len(hdr)

    for res in results:
        if not res.points:
            continue
        print(f"\n{res.name}")
        print(hdr); print(sep)
        for pt in res.points:
            print(
                f"{pt.n_vectors:>8,} | {pt.implementation:>6} | "
                f"{pt.ram_mb:>10.1f} | {pt.p50_ms:>10.3f} | "
                f"{pt.p99_ms:>10.3f} | {pt.recall_at_10:>10.3%} | "
                f"{pt.throughput_rps:>8.1f}"
            )

    print(f"\n{'='*76}")
    print("  COMPARAISON AU POINT MAXIMUM (données GloVe réelles)")
    print(f"{'='*76}")

    last = {res.name: res.points[-1] for res in results if res.points}
    if not last:
        return

    print(
        f"\n{'Engine':<50} {'Impl':>6} {'RAM':>8} "
        f"{'p50':>8} {'p99':>8} {'Recall':>8} {'RPS':>8}"
    )
    print("-" * 100)
    for name, pt in sorted(last.items(), key=lambda x: -x[1].recall_at_10):
        print(
            f"{name:<50} {pt.implementation:>6} "
            f"{pt.ram_mb:>7.1f}MB "
            f"{pt.p50_ms:>7.3f}ms "
            f"{pt.p99_ms:>7.3f}ms "
            f"{pt.recall_at_10:>7.2%} "
            f"{pt.throughput_rps:>7.0f}"
        )

    print(f"\n  Note : GloVe a une structure sémantique forte (clusters de mots).")
    print(f"  Le recall est bien plus élevé que sur des gaussiens — c'est réaliste.")
    print(f"  Note : latence Python ~10-20x plus haute que C++ par construction.")

    q    = last.get("Quorex (int8 + Piste1 + Piste2)")
    qopt = last.get("Quorex-Optimized (int8 + Piste1+2 + ef=80 + R=16)")
    hnsw = last.get("hnswlib (C++ float32)")
    faiss_f = last.get("FAISS IndexHNSWFlat (Meta, C++)")

    if q and qopt:
        gain = qopt.recall_at_10 - q.recall_at_10
        print(f"\n  Quorex vs Quorex-Optimized :")
        print(f"    Recall  : {q.recall_at_10:.2%} → {qopt.recall_at_10:.2%} ({gain:+.2%})")
        print(f"    Latence : {q.p50_ms:.3f}ms → {qopt.p50_ms:.3f}ms")

    if qopt and hnsw:
        print(f"\n  Quorex-Optimized vs hnswlib :")
        print(f"    Recall gap : {hnsw.recall_at_10 - qopt.recall_at_10:.2%} en faveur de hnswlib")
        print(f"    → gap comblé par portage Rust (ef élevé sans coût latence)")

# ── Export CSV ────────────────────────────────────────────────────────────────

def export_csv(results: list[EngineResult], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "engine", "implementation", "n_vectors", "ram_mb",
            "p50_ms", "p95_ms", "p99_ms", "recall_at_10", "throughput_rps"
        ])
        for res in results:
            for pt in res.points:
                w.writerow([
                    pt.engine, pt.implementation, pt.n_vectors,
                    round(pt.ram_mb, 2), round(pt.p50_ms, 4),
                    round(pt.p95_ms, 4), round(pt.p99_ms, 4),
                    round(pt.recall_at_10, 6), round(pt.throughput_rps, 2),
                ])
    print(f"\nRésultats exportés → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--glove",   type=str, required=True,
                   help="Chemin vers glove.6B.100d.txt")
    p.add_argument("--max-n",   type=int, default=100_000)
    p.add_argument("--steps",   type=int, default=4)
    p.add_argument("--k",       type=int, default=10)
    p.add_argument("--M",       type=int, default=16)
    p.add_argument("--ef",      type=int, default=50)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--csv",     type=str, default="results/benchmark_glove.csv")
    args = p.parse_args()

    results = run_suite(
        glove_path=args.glove,
        max_n=args.max_n,
        steps=args.steps,
        k=args.k,
        M=args.M,
        ef=args.ef,
        n_queries=args.queries,
    )
    print_report(results, k=args.k)
    export_csv(results, path=args.csv)