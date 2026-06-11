"""
Engine 3 : hnswlib float32 (C++)
Implémentation de référence HNSW en C++ par Malkov (l'auteur original).
Mêmes paramètres M et ef que Quorex pour comparaison algorithmique équitable.
Latence bien plus basse que Quorex Python — noté explicitement dans le rapport.
"""

from __future__ import annotations
import tracemalloc
import numpy as np
import hnswlib

from benchmarks.engines import BenchmarkEngine


class HnswlibEngine(BenchmarkEngine):
    name = "hnswlib (C++ float32)"

    def __init__(self, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index: hnswlib.Index | None = None
        self._peak_mb: float = 0.0

    def build(self, vecs: np.ndarray) -> None:
        n, dim = vecs.shape
        tracemalloc.start()

        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=n,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self._index.add_items(vecs, list(range(n)))
        self._index.set_ef(self.ef_search)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._peak_mb = peak / (1024 * 1024)

    def search(self, query: np.ndarray, top_k: int) -> list[int]:
        labels, _ = self._index.knn_query(query.reshape(1, -1), k=top_k)
        return labels[0].tolist()

    def ram_mb(self) -> float:
        return self._peak_mb

    def destroy(self) -> None:
        self._index = None
