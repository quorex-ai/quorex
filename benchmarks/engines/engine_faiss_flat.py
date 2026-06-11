"""
Engine 4 : FAISS IndexHNSWFlat (Meta, C++)
HNSW float32 de Meta — la référence académique.
Même algorithme que hnswlib mais implémentation Meta/FAIR.
Utilisé dans les papiers ANN comme baseline standard.
"""

from __future__ import annotations
import tracemalloc
import numpy as np
import faiss

from benchmarks.engines import BenchmarkEngine


class FaissHNSWFlatEngine(BenchmarkEngine):
    name = "FAISS IndexHNSWFlat (Meta, C++)"

    def __init__(self, M: int = 16, ef_search: int = 50):
        self.M = M
        self.ef_search = ef_search
        self._index: faiss.IndexHNSWFlat | None = None
        self._peak_mb: float = 0.0

    def build(self, vecs: np.ndarray) -> None:
        n, dim = vecs.shape
        tracemalloc.start()

        # IndexHNSWFlat utilise le produit interne — on normalise les vecteurs
        # pour que le produit interne soit équivalent à la similarité cosine.
        self._index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = 200
        self._index.hnsw.efSearch       = self.ef_search
        self._index.add(vecs.astype(np.float32))

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._peak_mb = peak / (1024 * 1024)

    def search(self, query: np.ndarray, top_k: int) -> list[int]:
        q = query.reshape(1, -1).astype(np.float32)
        _, labels = self._index.search(q, top_k)
        return labels[0].tolist()

    def ram_mb(self) -> float:
        return self._peak_mb

    def destroy(self) -> None:
        self._index = None
