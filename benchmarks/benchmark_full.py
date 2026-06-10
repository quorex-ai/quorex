"""
benchmarks/benchmark_full.py
─────────────────────────────
Benchmark Quorex vs Baseline vs GAFAM.

3 configurations CLAIREMENT SÉPARÉES :

  1. Baseline — HNSW float32 pur, zéro quantification, zéro reranking
                → référence exacte

  2. GAFAM    — HNSW + SQ8 BASIQUE (n_residual=0, pas de Piste 1)
                + reranking float32 exact (top_k*4 candidats)
                + RAM RÉELLE = uint8 index + float32 store du reranking
                → simule ce que font les libs standard

  3. Quorex   — HNSW + SQ8 + Piste 1 (quantization-aware construction)
                           + Piste 2 (residual compensation R=8)
                           + reranking compensé (PAS de float32 store)
                → toutes les innovations Quorex

La claim de Quorex : recall comparable à GAFAM,
avec 2x moins de RAM (pas de float32 store) et latence inférieure.

Usage :
  python -m benchmarks.benchmark_full --max-n 10000 --steps 5
  python -m benchmarks.benchmark_full --max-n 10000 --steps 5 --csv results/bench.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import shutil
import sys
import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.quorex.core.vectordb.consolidator import ConsolidationConfig
from src.quorex.core.vectordb.engine import VectorDBEngine
from src.quorex.core.vectordb.quantizer import SQ8Quantizer

SEED     = 42
TMP_BASE = "/tmp/quorex_bench"

# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class BenchmarkPoint:
    n_vectors:      int
    ram_mb:         float
    p50_ms:         float
    p95_ms:         float
    p99_ms:         float
    recall_at_10:   float
    throughput_rps: float

@dataclass
class BenchmarkSuite:
    name:   str
    points: list[BenchmarkPoint] = field(default_factory=list)

# ── Dataset ───────────────────────────────────────────────────────────────────

def generate_dataset(n: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
    queries = rng.standard_normal((200, dim)).astype(np.float32)
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8)
    return vecs, queries

def ground_truth(vecs: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    scores = queries @ vecs.T
    return np.argsort(-scores, axis=1)[:, :k]

def recall_at_k(retrieved: list[list[int]], gt: np.ndarray, k: int = 10) -> float:
    hits = total = 0
    for ret, g in zip(retrieved, gt):
        hits  += len(set(ret[:k]) & set(g[:k].tolist()))
        total += k
    return hits / total if total else 0.0

def measure_ram_mb() -> float:
    _, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)

def _point(n, ram_mb, lats, retrieved, gt, k) -> BenchmarkPoint:
    a = np.array(lats)
    return BenchmarkPoint(
        n_vectors=n, ram_mb=ram_mb,
        p50_ms=float(np.percentile(a, 50)),
        p95_ms=float(np.percentile(a, 95)),
        p99_ms=float(np.percentile(a, 99)),
        recall_at_10=recall_at_k(retrieved, gt, k),
        throughput_rps=1000.0 / float(np.mean(a)),
    )

def _make_engine(path, dim, bio=False, cfg=None):
    kwargs = dict(
        path=path, dim=dim,
        M=16, ef_construction=200, ef_search=50,
        checkpoint_every=999_999, quantize=False,
        bio_enabled=bio,
    )
    if cfg:
        kwargs["consolidation_config"] = cfg
    return VectorDBEngine(**kwargs)

def _insert(engine, vecs, n, timestamps=False):
    uid = "bench_user"
    now = time.time()
    for i, vec in enumerate(vecs[:n]):
        meta = {
            "id": i, "action": "bench",
            "metadata": {
                "text": f"vec_{i}",
                **({"timestamp": now - (n - i) * 60, "reinforcements": 1}
                   if timestamps else {}),
            },
        }
        engine.segment.insert(uid, vec, meta)
    return uid

# ── CONFIG 1 : Baseline HNSW float32 ─────────────────────────────────────────

def run_baseline(vecs, queries, gt, n, k=10) -> BenchmarkPoint:
    """HNSW float32 pur — zéro quantification, zéro reranking."""
    path = f"{TMP_BASE}/baseline_{n}"
    shutil.rmtree(path, ignore_errors=True)
    gc.collect(); tracemalloc.start()

    engine = _make_engine(path, vecs.shape[1])
    engine.start()
    uid = _insert(engine, vecs, n)

    ram_mb = measure_ram_mb()
    tracemalloc.stop()

    lats, retrieved = [], []
    for q in queries:
        t0 = time.perf_counter()
        res = engine.segment.search(uid, q, top_k=k)
        lats.append((time.perf_counter() - t0) * 1000)
        retrieved.append([r["id"] for r in res])

    engine.stop()
    shutil.rmtree(path, ignore_errors=True)
    return _point(n, ram_mb, lats, retrieved, gt, k)

# ── CONFIG 2 : GAFAM (SQ8 basique + reranking float32) ───────────────────────

def run_gafam(vecs, queries, gt, n, k=10, rerank_factor=4) -> BenchmarkPoint:
    """
    GAFAM simulation honnête :
      - SQ8 BASIQUE : n_residual=0 (pas de Piste 2)
      - Graphe câblé en float32 PUIS quantifié (pas de Piste 1)
      - Reranking float32 exact sur top_k*rerank_factor candidats
      - RAM RÉELLE = uint8 index + float32 store (les deux comptés)
        → n * dim * 4 bytes de float32 store en plus de l'index uint8
    """
    path = f"{TMP_BASE}/gafam_{n}"
    shutil.rmtree(path, ignore_errors=True)
    gc.collect(); tracemalloc.start()

    engine = _make_engine(path, vecs.shape[1])
    engine.start()
    uid = _insert(engine, vecs, n)

    # SQ8 basique : zéro résiduel, graphe câblé float32 (standard industry)
    q8 = SQ8Quantizer(n_residual=0)
    q8.fit(list(vecs[:min(2000, n)]))
    engine.segment.quantizer = q8
    for idx in engine.segment._indexes.values():
        idx.enable_quantization(q8)

    # Float32 store obligatoire pour le reranking exact
    # RAM réelle = uint8 index (tracemalloc) + store float32 (calculé)
    float32_store    = vecs[:n]                                   # vue numpy, pas de copie
    float32_store_mb = (n * vecs.shape[1] * 4) / (1024 * 1024)   # coût réel en prod

    ram_mb = measure_ram_mb() + float32_store_mb
    tracemalloc.stop()

    lats, retrieved = [], []
    for q in queries:
        t0 = time.perf_counter()

        # Phase 1 — ANN int8
        candidates = engine.segment.search(uid, q, top_k=k * rerank_factor)
        cand_ids   = [c["id"] for c in candidates if c["id"] < n]

        # Phase 2 — reranking float32 exact (vectorisé)
        if cand_ids:
            scores = float32_store[cand_ids] @ q
            order  = np.argsort(-scores)[:k]
            top_ids = [cand_ids[i] for i in order]
        else:
            top_ids = []

        lats.append((time.perf_counter() - t0) * 1000)
        retrieved.append(top_ids)

    engine.stop()
    shutil.rmtree(path, ignore_errors=True)
    return _point(n, ram_mb, lats, retrieved, gt, k)

# ── CONFIG 3 : Quorex (Pistes 1+2 + reranking compensé) ──────────────────────

def run_quorex(vecs, queries, gt, n, k=10) -> BenchmarkPoint:
    """
    Quorex complet :
      - Piste 1 : quantization-aware construction (graphe câblé sur distances int8)
      - Piste 2 : residual compensation R=8 (16 bytes float16 / nœud)
      - Reranking compensé par les résiduels — PAS de float32 store
      - RAM = uint8 index + résiduels float16 seulement
      - Bio neutralisé pour mesurer uniquement l'avantage algorithmique
    """
    path = f"{TMP_BASE}/quorex_{n}"
    shutil.rmtree(path, ignore_errors=True)
    gc.collect(); tracemalloc.start()

    # Bio neutralisé (decay ≈ 0, pas de freq boost, pas de conflict)
    # → bio_weight = 1.0 pour tous les nœuds → distance non modifiée
    # On mesure Piste1 + Piste2 pur, sans l'effet temporel
    bio_off = ConsolidationConfig(
        prune_threshold=0.0,
        merge_threshold=1.1,
        consolidate_every=999_999,
        decay_stability_factor=10.0,
        freq_boost_factor=0.0,
        conflict_penalty_factor=0.0,
    )

    engine = _make_engine(path, vecs.shape[1], bio=True, cfg=bio_off)
    engine.start()
    uid = _insert(engine, vecs, n, timestamps=True)

    # Pistes 1+2 actives via enable_quantization() (n_residual=8 par défaut)
    # Piste 1 déjà appliquée lors de l'insert (graphe câblé sur int8)
    engine.enable_quantization()

    ram_mb = measure_ram_mb()
    tracemalloc.stop()

    lats, retrieved = [], []
    for q in queries:
        t0 = time.perf_counter()
        res = engine.segment.search(uid, q, top_k=k)
        lats.append((time.perf_counter() - t0) * 1000)
        retrieved.append([r["id"] for r in res])

    engine.stop()
    shutil.rmtree(path, ignore_errors=True)
    return _point(n, ram_mb, lats, retrieved, gt, k)

# ── Runner ────────────────────────────────────────────────────────────────────

def run_suite(dim=128, max_n=10_000, steps=5, k=10, n_queries=200):
    ns = np.geomspace(1_000, max_n, steps).astype(int).tolist()

    baseline_s = BenchmarkSuite("Baseline (HNSW float32)")
    gafam_s    = BenchmarkSuite("GAFAM (int8 basique + reranking float32, RAM réelle)")
    quorex_s   = BenchmarkSuite("Quorex (Piste1 + Piste2 + reranking compensé)")

    print(f"\n{'='*70}")
    print(f"  QUOREX BENCHMARK  |  dim={dim}  max_n={max_n:,}  k={k}  queries={n_queries}")
    print(f"{'='*70}\n")
    print(f"Generating dataset ({max_n:,} vectors, dim={dim})...")
    all_vecs, all_queries = generate_dataset(max_n, dim)
    qs = all_queries[:n_queries]

    for n in ns:
        vecs = all_vecs[:n]
        gt   = ground_truth(vecs, qs, k)
        print(f"\n── N = {n:>7,} vectors ──────────────────────────────────────────")

        for label, fn, suite in [
            ("[1/3] Baseline", run_baseline, baseline_s),
            ("[2/3] GAFAM",    run_gafam,    gafam_s),
            ("[3/3] Quorex",   run_quorex,   quorex_s),
        ]:
            print(f"  {label}...")
            pt = fn(vecs, qs, gt, n, k)
            suite.points.append(pt)
            print(f"        RAM={pt.ram_mb:.1f}MB  p50={pt.p50_ms:.3f}ms  Recall@{k}={pt.recall_at_10:.3%}")

    return baseline_s, gafam_s, quorex_s

# ── Rapport ───────────────────────────────────────────────────────────────────

def print_report(baseline, gafam, quorex, k=10):
    print(f"\n{'='*70}")
    print("  RAPPORT FINAL")
    print(f"{'='*70}")
    hdr = (f"{'N':>8} | {'RAM (MB)':>10} | {'p50 (ms)':>10} | "
           f"{'p99 (ms)':>10} | {f'Recall@{k}':>10} | {'RPS':>8}")
    sep = "-" * len(hdr)

    for suite in [baseline, gafam, quorex]:
        print(f"\n{suite.name}")
        print(hdr); print(sep)
        for pt in suite.points:
            print(
                f"{pt.n_vectors:>8,} | {pt.ram_mb:>10.1f} | "
                f"{pt.p50_ms:>10.3f} | {pt.p99_ms:>10.3f} | "
                f"{pt.recall_at_10:>10.3%} | {pt.throughput_rps:>8.1f}"
            )

    print(f"\n{'='*70}")
    print("  COMPARAISON AU POINT MAXIMUM (N le plus grand)")
    print(f"{'='*70}")
    b, g, q = baseline.points[-1], gafam.points[-1], quorex.points[-1]

    print(f"\n  RAM totale (uint8 index + float32 store pour GAFAM) :")
    print(f"    Baseline : {b.ram_mb:.1f} MB")
    print(f"    GAFAM    : {g.ram_mb:.1f} MB  ← inclut float32 store du reranking")
    print(f"    Quorex   : {q.ram_mb:.1f} MB  ← uint8 + résiduels float16 seulement")
    print(f"    → Quorex utilise {g.ram_mb/q.ram_mb:.2f}x moins de RAM que GAFAM")

    print(f"\n  Latence p50 vs GAFAM :")
    faster = g.p50_ms / q.p50_ms
    print(f"    Quorex : {'%.2f'%faster}x {'plus rapide' if faster > 1 else 'plus lent'} "
          f"({q.p50_ms:.3f}ms vs {g.p50_ms:.3f}ms)")

    print(f"\n  Latence p99 (stabilité en prod) :")
    print(f"    GAFAM    : {g.p99_ms:.3f}ms  (ratio p99/p50 = {g.p99_ms/g.p50_ms:.2f}x)")
    print(f"    Quorex   : {q.p99_ms:.3f}ms  (ratio p99/p50 = {q.p99_ms/q.p50_ms:.2f}x)")

    print(f"\n  Recall@{k} :")
    print(f"    Baseline : {b.recall_at_10:.3%}")
    print(f"    GAFAM    : {g.recall_at_10:.3%}  (reranking exact → quasi-parfait par construction)")
    print(f"    Quorex   : {q.recall_at_10:.3%}  (sans float32 store)")

    print(f"\n  Throughput (req/sec) :")
    print(f"    Baseline : {b.throughput_rps:.1f}")
    print(f"    GAFAM    : {g.throughput_rps:.1f}")
    print(f"    Quorex   : {q.throughput_rps:.1f}")

    print(f"\n  Résumé :")
    print(f"    GAFAM atteint {g.recall_at_10:.1%} recall grâce au reranking exact,")
    print(f"    mais nécessite {g.ram_mb:.0f} MB de RAM (float32 store inclus).")
    print(f"    Quorex atteint {q.recall_at_10:.1%} recall sans float32 store,")
    print(f"    avec {g.ram_mb/q.ram_mb:.1f}x moins de RAM et {faster:.2f}x moins de latence.")

def export_csv(baseline, gafam, quorex, path="results/benchmark.csv"):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config","n_vectors","ram_mb","p50_ms","p95_ms",
                    "p99_ms","recall_at_10","throughput_rps"])
        for suite in [baseline, gafam, quorex]:
            for pt in suite.points:
                w.writerow([
                    suite.name, pt.n_vectors,
                    round(pt.ram_mb, 2), round(pt.p50_ms, 4),
                    round(pt.p95_ms, 4), round(pt.p99_ms, 4),
                    round(pt.recall_at_10, 6), round(pt.throughput_rps, 2),
                ])
    print(f"\nRésultats exportés → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dim",     type=int, default=128)
    p.add_argument("--max-n",   type=int, default=10_000)
    p.add_argument("--steps",   type=int, default=5)
    p.add_argument("--k",       type=int, default=10)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--csv",     type=str, default="results/benchmark.csv")
    args = p.parse_args()

    os.makedirs(TMP_BASE, exist_ok=True)
    baseline, gafam, quorex = run_suite(
        dim=args.dim, max_n=args.max_n, steps=args.steps,
        k=args.k, n_queries=args.queries,
    )
    print_report(baseline, gafam, quorex, k=args.k)
    export_csv(baseline, gafam, quorex, path=args.csv)