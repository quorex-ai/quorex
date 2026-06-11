"""
Engine 1 : Quorex HNSW float32 pur
Référence exacte — zéro quantification, zéro reranking.
Équivalent à FAISS IndexHNSWFlat mais en Python.
"""

from __future__ import annotations
import os, sys, time, tracemalloc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.quorex.core.vectordb.hnsw import HNSWIndex
from benchmarks.engines import BenchmarkEngine


class QuorexBaselineEngine(BenchmarkEngine):
    name = "Quorex-Baseline (HNSW float32)"

    def __init__(self, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index: HNSWIndex | None = None
        self._peak_mb: float = 0.0

    def build(self, vecs: np.ndarray) -> None:
        tracemalloc.start()
        self._index = HNSWIndex(
            dim=vecs.shape[1],
            M=self.M,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
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
