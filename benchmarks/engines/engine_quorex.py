"""
Engine 2 : Quorex int8 + Piste1 + Piste2
- Piste 1 : quantization-aware construction (graphe câblé sur int8)
- Piste 2 : residual compensation R=8 (float16 par nœud)
- Reranking compensé — PAS de float32 store séparé

Deux configs exposées :
  QuorexEngine          — ef=50, R=8  (config de base)
  QuorexOptimizedEngine — ef=80, R=16 (config optimisée, recall ↑, latence +~30%)
"""

from __future__ import annotations
import os, sys, tracemalloc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.quorex.core.vectordb.hnsw import HNSWIndex
from src.quorex.core.vectordb.quantizer import SQ8Quantizer
from benchmarks.engines import BenchmarkEngine


class QuorexEngine(BenchmarkEngine):
    name = "Quorex (int8 + Piste1 + Piste2)"

    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        n_residual: int = 8,
        fit_sample: int = 2000,
    ):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.n_residual = n_residual
        self.fit_sample = fit_sample
        self._index: HNSWIndex | None = None
        self._peak_mb: float = 0.0

    def build(self, vecs: np.ndarray) -> None:
        tracemalloc.start()

        q = SQ8Quantizer(n_residual=self.n_residual)
        q.fit(list(vecs[:min(self.fit_sample, len(vecs))]))

        self._index = HNSWIndex(
            dim=vecs.shape[1],
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            quantizer=q,
            bio_enabled=False,
        )
        for i, v in enumerate(vecs):
            self._index.insert(i, v)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._peak_mb = peak / (1024 * 1024)

    def search(self, query: np.ndarray, top_k: int) -> list[int]:
        results = self._index.search(query, top_k=top_k)
        return [nid for nid, _ in results]

    def ram_mb(self) -> float:
        return self._peak_mb

    def destroy(self) -> None:
        self._index = None


class QuorexOptimizedEngine(QuorexEngine):
    """
    Config optimisée :
      - ef_search=80  → explore 60% plus de candidats → recall ↑
      - n_residual=16 → compensation résiduelle plus précise → recall ↑
      - Coût : latence +30% environ, RAM +~1.5MB vs config de base
      - Gain estimé : +4 à 6 points de recall vs QuorexEngine
    """
    name = "Quorex-Optimized (int8 + Piste1+2 + ef=80 + R=16)"

    def __init__(self, M: int = 16, ef_construction: int = 200, fit_sample: int = 2000):
        super().__init__(
            M=M,
            ef_construction=ef_construction,
            ef_search=80,
            n_residual=16,
            fit_sample=fit_sample,
        )