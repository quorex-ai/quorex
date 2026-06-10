"""
quorex.core.vectordb.quantizer
-------------------------------
Scalar quantization 8-bit (SQ8): float32 → uint8.
4x memory reduction with minimal recall loss.

Factored asymmetric distance (the key optimization):
  dot(q, decode(u)) = dot(q, u*scale + offset)
                    = dot(q*scale, u) + dot(q, offset)
                    = dot(q_scaled, u) + q_offset

  q_scaled and q_offset are precomputed ONCE per query, so each per-node
  distance is a single dot(q_scaled, u) over uint8 — no per-node decode,
  no per-node float32 allocation. This is the FAISS trick.

Asymmetric quantization:
  - Query vectors stay float32 (no loss on the query side)
  - Stored vectors are uint8
"""

from __future__ import annotations

import os
import numpy as np


class SQ8Quantizer:
    def __init__(self, n_residual: int = 8) -> None:
        self._scale:  np.ndarray | None = None
        self._offset: np.ndarray | None = None
        self._fitted: bool = False
        # Residual error compensation (Piste 2)
        self.n_residual = n_residual
        self._residual_basis: np.ndarray | None = None   # (R, dim) float32

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, vectors: list[np.ndarray]) -> None:
        if not vectors:
            raise ValueError("Cannot fit on empty vector list.")

        matrix = np.stack(vectors, axis=0).astype(np.float32)
        lo = matrix.min(axis=0)
        hi = matrix.max(axis=0)

        diff = hi - lo
        diff[diff == 0.0] = 1.0

        self._scale  = (diff / 255.0).astype(np.float32)
        self._offset = lo.astype(np.float32)
        self._fitted = True

        # Residual basis on quantization errors — captures the principal
        # directions of SQ8 error so the compensated distance can recover
        # most of the quantization loss.
        decoded = self.decode_batch(self.encode_batch(matrix))
        errors = matrix - decoded
        sample = errors[:2000]
        _, _, Vt = np.linalg.svd(sample, full_matrices=False)
        self._residual_basis = Vt[: self.n_residual].astype(np.float32)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, vector: np.ndarray) -> np.ndarray:
        self._check_fitted()
        v = vector.astype(np.float32)
        q = (v - self._offset) / self._scale
        return np.clip(np.round(q), 0, 255).astype(np.uint8)

    def decode(self, quantized: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return quantized.astype(np.float32) * self._scale + self._offset

    def encode_batch(self, vectors) -> np.ndarray:
        self._check_fitted()
        m = (
            np.stack(vectors, axis=0).astype(np.float32)
            if not isinstance(vectors, np.ndarray)
            else vectors.astype(np.float32)
        )
        q = (m - self._offset) / self._scale
        return np.clip(np.round(q), 0, 255).astype(np.uint8)

    def decode_batch(self, quantized: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return quantized.astype(np.float32) * self._scale + self._offset

    # ------------------------------------------------------------------
    # FACTORED query preparation — the core optimization
    # ------------------------------------------------------------------

    def prepare_query(self, query_f32: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Precompute the factored query terms ONCE per search.

        Returns:
          q_scaled : query * scale            (float32, dim)  → dotted with uint8
          q_offset : dot(query, offset)       (float scalar)  → constant per query

        Then for any stored uint8 vector u:
          dot(query, decode(u)) = dot(q_scaled, u) + q_offset
        No decode, no per-node allocation.
        """
        self._check_fitted()
        q = query_f32.astype(np.float32)
        q_scaled = q * self._scale
        q_offset = float(np.dot(q, self._offset))
        return q_scaled, q_offset

    def factored_distance(
        self,
        u: np.ndarray,
        q_scaled: np.ndarray,
        q_offset: float,
    ) -> float:
        """Cosine distance for one uint8 node via the factored form."""
        dot = float(u @ q_scaled) + q_offset
        return 1.0 - dot

    def factored_distance_batch(
        self,
        U: np.ndarray,          # (k, dim) uint8
        q_scaled: np.ndarray,   # (dim,) float32
        q_offset: float,
    ) -> np.ndarray:
        """Cosine distance for a batch of uint8 nodes — one matmul, no decode."""
        dots = U.astype(np.float32) @ q_scaled + q_offset   # (k,)
        return 1.0 - dots

    # ------------------------------------------------------------------
    # Residual compensation (Piste 2)
    # ------------------------------------------------------------------

    def encode_residual(self, vector: np.ndarray) -> np.ndarray:
        self._check_fitted()
        v = vector.astype(np.float32)
        decoded = self.decode(self.encode(v))
        e = v - decoded
        return (self._residual_basis @ e).astype(np.float16)

    def project_query(self, query_f32: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (self._residual_basis @ query_f32.astype(np.float32))

    # ------------------------------------------------------------------
    # Legacy non-factored distances (kept for _prune / greedy fallback)
    # ------------------------------------------------------------------

    def asymmetric_distance(self, query_f32, stored_uint8) -> float:
        self._check_fitted()
        decoded = stored_uint8.astype(np.float32) * self._scale + self._offset
        return float(np.dot(query_f32, decoded))

    def compensated_distance(
        self,
        query_f32: np.ndarray,
        stored_uint8: np.ndarray,
        residual_f16: np.ndarray,
        q_proj: np.ndarray,
    ) -> float:
        base = self.asymmetric_distance(query_f32, stored_uint8)
        corr = float(np.dot(q_proj, residual_f16.astype(np.float32)))
        return base + corr

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        self._check_fitted()
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez(
            path,
            scale=self._scale,
            offset=self._offset,
            residual_basis=self._residual_basis,
        )

    def load(self, path: str) -> None:
        data = np.load(path + ".npz")
        self._scale  = data["scale"].astype(np.float32)
        self._offset = data["offset"].astype(np.float32)
        if "residual_basis" in data:
            self._residual_basis = data["residual_basis"].astype(np.float32)
        self._fitted = True

    @property
    def compression_ratio(self) -> float:
        return 4.0

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SQ8Quantizer must be fitted before use.")

    def __repr__(self) -> str:
        if not self._fitted:
            return "SQ8Quantizer(not fitted)"
        dim = len(self._scale)
        return f"SQ8Quantizer(dim={dim}, compression=4x, residual_R={self.n_residual})"


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dim = 64

    vecs = [rng.standard_normal(dim).astype(np.float32) for _ in range(500)]
    vecs = [v / np.linalg.norm(v) for v in vecs]

    q = SQ8Quantizer()
    q.fit(vecs)
    print(q)

    v = vecs[0]
    enc = q.encode(v)

    # Verify factored distance == naive decode distance
    q_scaled, q_offset = q.prepare_query(v)
    factored = q.factored_distance(enc, q_scaled, q_offset)
    naive = 1.0 - q.asymmetric_distance(v, enc)
    print(f"Factored distance : {factored:.6f}")
    print(f"Naive distance    : {naive:.6f}")
    print(f"Difference        : {abs(factored - naive):.2e}  (should be ~0)")