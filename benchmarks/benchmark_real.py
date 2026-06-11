"""
benchmarks/benchmark_real.py
─────────────────────────────
Benchmark multi-engine : Quorex vs hnswlib vs FAISS.

Engines comparés :
  1. Quorex-Baseline  — HNSW float32 Python (référence)
  2. Quorex           — HNSW int8 + Piste1 + Piste2 (notre contribution)
  3. hnswlib           — HNSW float32 C++ (auteur original, référence C++)
  4. FAISS HNSWFlat   — HNSW float32 Meta (référence académique)
  5. FAISS IVF+SQ8    — IVF + SQ8 Meta (approche industrie)

Tous les engines utilisent M=16, ef=50 pour une comparaison algorithmique équitable.
La latence C++ vs Python est notée explicitement — pas de prétention à gagner sur la
vitesse brute en Python pur.

Usage :
  python -m benchmarks.benchmark_real --max-n 10000 --steps 5
  python -m benchmarks.benchmark_real --max-n 10000 --steps 5 --csv results/benchmark_real.csv
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
from benchmarks.engines.engine_quorex         import QuorexEngine
from benchmarks.engines.engine_hnswlib         import HnswlibEngine
from benchmarks.engines.engine_faiss_flat      import FaissHNSWFlatEngine
from benchmarks.engines.engine_faiss_sq8       import FaissIVFSQ8Engine

SEED     = 42
TMP_BASE = "/tmp/quorex_bench"

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
    implementation: str   # "Python" ou "C++"

@dataclass
class EngineResult:
    name:   str
    points: list[BenchmarkPoint] = field(default_factory=list)

# ── Dataset ───────────────────────────────────────────────────────────────────

def generate_dataset(n: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    n vecteurs normalisés + 200 requêtes tirées du corpus.
    Garantit de vrais voisins à retrouver pour un recall correct.
    """
    rng = np.random.default_rng(SEED)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)

    n_queries = min(200, n)
    rng_q     = np.random.default_rng(SEED + 1)
    idx       = rng_q.choice(n, size=n_queries, replace=False)
    queries   = vecs[idx].copy()

    return vecs, queries

def ground_truth(vecs: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    scores = queries @ vecs.T
    return np.argsort(-scores, axis=1)[:, :k]

def recall_at_k(
    retrieved: list[list[int]], gt: np.ndarray, k: int = 10
) -> float:
    hits = total = 0
    for ret, g in zip(retrieved, gt):
        hits  += len(set(ret[:k]) & set(g[:k].tolist()))
        total += k
    return hits / total if total else 0.0

# ── Benchmark d'un engine ─────────────────────────────────────────────────────

def bench_engine(
    engine,
    vecs: np.ndarray,
    queries: np.ndarray,
    gt: np.ndarray,
    n: int,
    k: int = 10,
    impl: str = "Python",
) -> BenchmarkPoint:
    gc.collect()

    # Build
    engine.build(vecs[:n])
    ram = engine.ram_mb()

    # Search
    lats, retrieved = [], []
    for q in queries:
        t0 = time.perf_counter()
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
    dim: int = 128,
    max_n: int = 10_000,
    steps: int = 5,
    k: int = 10,
    M: int = 16,
    ef: int = 50,
) -> list[EngineResult]:

    ns = np.geomspace(1_000, max_n, steps).astype(int).tolist()

    # Définit les engines avec les mêmes paramètres M et ef
    engines_cfg = [
        (QuorexBaselineEngine(M=M, ef_construction=200, ef_search=ef), "Python"),
        (QuorexEngine(M=M, ef_construction=200, ef_search=ef),         "Python"),
        (HnswlibEngine(M=M, ef_construction=200, ef_search=ef),        "C++"),
        (FaissHNSWFlatEngine(M=M, ef_search=ef),                       "C++"),
        (FaissIVFSQ8Engine(nlist=max(4, min(100, max_n//10)), nprobe=10), "C++"),
    ]

    results = {e.name: EngineResult(name=e.name) for e, _ in engines_cfg}

    print(f"\n{'='*76}")
    print(
        f"  BENCHMARK RÉEL  |  dim={dim}  max_n={max_n:,}  "
        f"k={k}  M={M}  ef={ef}"
    )
    print(f"{'='*76}\n")
    print("Engines :")
    for e, impl in engines_cfg:
        print(f"  [{impl:6}] {e.name}")

    for n in ns:
        vecs, queries = generate_dataset(n, dim)
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
    print("  RAPPORT FINAL — COMPARAISON MULTI-ENGINE")
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

    # Comparaison au point max
    print(f"\n{'='*76}")
    print("  COMPARAISON AU POINT MAXIMUM")
    print(f"{'='*76}")

    last = {
        res.name: res.points[-1]
        for res in results
        if res.points
    }
    if not last:
        return

    print(f"\n{'Engine':<45} {'Impl':>6} {'RAM':>8} {'p50':>8} {'p99':>8} {'Recall':>8} {'RPS':>8}")
    print("-" * 95)
    for name, pt in sorted(last.items(), key=lambda x: -x[1].recall_at_10):
        print(
            f"{name:<45} {pt.implementation:>6} "
            f"{pt.ram_mb:>7.1f}MB "
            f"{pt.p50_ms:>7.3f}ms "
            f"{pt.p99_ms:>7.3f}ms "
            f"{pt.recall_at_10:>7.2%} "
            f"{pt.throughput_rps:>7.0f}"
        )

    print(f"\n  Note: la latence Python est ~10-20x plus haute que C++ par construction.")
    print(f"  La comparaison équitable est le RECALL et la RAM à M et ef identiques.")

    # Trouve Quorex et hnswlib pour la comparaison recall/RAM
    quorex = last.get("Quorex (int8 + Piste1 + Piste2)")
    hnswlib_e = last.get("hnswlib (C++ float32)")
    faiss_flat = last.get("FAISS IndexHNSWFlat (Meta, C++)")

    if quorex and hnswlib_e:
        print(f"\n  Quorex vs hnswlib (même algo, Python vs C++) :")
        print(f"    Recall  : Quorex {quorex.recall_at_10:.2%} vs hnswlib {hnswlib_e.recall_at_10:.2%}")
        print(f"    RAM     : Quorex {quorex.ram_mb:.1f}MB vs hnswlib {hnswlib_e.ram_mb:.1f}MB")
        print(f"    Latence : Quorex {quorex.p50_ms:.3f}ms vs hnswlib {hnswlib_e.p50_ms:.3f}ms (C++ avantage attendu)")

    if quorex and faiss_flat:
        print(f"\n  Quorex vs FAISS HNSWFlat (référence académique) :")
        print(f"    Recall  : Quorex {quorex.recall_at_10:.2%} vs FAISS {faiss_flat.recall_at_10:.2%}")
        print(f"    RAM     : Quorex {quorex.ram_mb:.1f}MB vs FAISS {faiss_flat.ram_mb:.1f}MB")

# ── Export CSV ────────────────────────────────────────────────────────────────

def export_csv(results: list[EngineResult], path: str = "results/benchmark_real.csv") -> None:
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
    p.add_argument("--dim",    type=int, default=128)
    p.add_argument("--max-n",  type=int, default=10_000)
    p.add_argument("--steps",  type=int, default=5)
    p.add_argument("--k",      type=int, default=10)
    p.add_argument("--M",      type=int, default=16)
    p.add_argument("--ef",     type=int, default=50)
    p.add_argument("--csv",    type=str, default="results/benchmark_real.csv")
    args = p.parse_args()

    os.makedirs(TMP_BASE, exist_ok=True)
    results = run_suite(
        dim=args.dim, max_n=args.max_n, steps=args.steps,
        k=args.k, M=args.M, ef=args.ef,
    )
    print_report(results, k=args.k)
    export_csv(results, path=args.csv)
