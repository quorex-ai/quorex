"""
Engine 5 : FAISS IndexIVFScalarQuantizer (Meta, C++)
IVF + SQ8 — ce que font vraiment les GAFAM en prod à grande échelle.
Approche industrielle : clustering IVF + quantification scalaire 8-bit.

RAM réelle = index uint8 + centroids float32 (comptés ensemble).
Nécessite un entraînement (fit) sur un échantillon avant l'indexation.

Note : nécessite nlist vecteurs minimum pour l'entraînement.
       On adapte nlist automatiquement selon n.
"""

from __future__ import annotations
import tracemalloc
import numpy as np
import faiss

from benchmarks.engines import BenchmarkEngine


class FaissIVFSQ8Engine(BenchmarkEngine):
    name = "FAISS IVF+SQ8 (Meta, C++)"

    def __init__(self, nlist: int = 100, nprobe: int = 10):
        self.nlist  = nlist
        self.nprobe = nprobe
        self._index: faiss.IndexIVFScalarQuantizer | None = None
        self._peak_mb: float = 0.0

    def build(self, vecs: np.ndarray) -> None:
        n, dim = vecs.shape
        tracemalloc.start()

        # Adapte nlist si n est petit
        nlist = min(self.nlist, max(1, n // 10))

        quantizer = faiss.IndexFlatIP(dim)
        self._index = faiss.IndexIVFScalarQuantizer(
            quantizer, dim, nlist,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT,
        )
        # Entraînement sur les vecteurs
        train_vecs = vecs[:min(50_000, n)].astype(np.float32)
        self._index.train(train_vecs)
        self._index.add(vecs.astype(np.float32))
        self._index.nprobe = min(self.nprobe, nlist)

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
