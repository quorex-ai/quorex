"""
benchmarks/engines
------------------
Interface commune pour tous les engines benchmarkés.

Chaque engine expose :
  - name        : str
  - build()     : construit l'index sur les vecteurs fournis
  - search()    : retourne les top_k ids les plus proches
  - ram_mb()    : RAM consommée par l'index (hors float32 store si applicable)
  - destroy()   : libère les ressources
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BenchmarkEngine(ABC):
    name: str = "unnamed"

    @abstractmethod
    def build(self, vecs: np.ndarray) -> None:
        """Construit l'index sur les vecteurs (n, dim) float32 normalisés."""
        ...

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int) -> list[int]:
        """Retourne les top_k ids les plus proches de la requête."""
        ...

    @abstractmethod
    def ram_mb(self) -> float:
        """RAM consommée par l'index en MB."""
        ...

    def destroy(self) -> None:
        """Libère les ressources (optionnel)."""
        pass
