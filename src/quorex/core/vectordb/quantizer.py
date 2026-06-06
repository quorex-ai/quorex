"""
quorex.core.vectordb.quantizer
-------------------------------
Scalar quantization 8-bit (SQ8): float32 → uint8.
4x memory reduction with minimal recall loss.

Formula (per dimension d):
  scale[d]  = (max[d] - min[d]) / 255
  offset[d] = min[d]
  q[d]      = clip(round((x[d] - offset[d]) / scale[d]), 0, 255).astype(uint8)
  x̂[d]      = q[d] * scale[d] + offset[d]   ← reconstruction

Asymmetric quantization:
  - Query vectors stay float32 (no loss on the query side)
  - Stored vectors are uint8
  - Distance ≈ 1 − dot(query_f32, decode(stored_uint8))
"""

from __future__ import annotations

import os
import numpy as np


class SQ8Quantizer:
    def __init__(self) -> None:
        self._scale:  np.ndarray | None = None   # (dim,) float32
        self._offset: np.ndarray | None = None   # (dim,) float32 — per-dim min
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, vectors: list[np.ndarray]) -> None:
        """
        Learns per-dimension min/max from a sample of float32 vectors.
        All vectors must have the same dimensionality.
        """
        if not vectors:
            raise ValueError("Cannot fit on empty vector list.")

        matrix = np.stack(vectors, axis=0).astype(np.float32)  # (n, dim)
        lo = matrix.min(axis=0)   # (dim,)
        hi = matrix.max(axis=0)   # (dim,)

        diff = hi - lo
        # Prevent division by zero for constant dimensions (assign scale=1 so
        # encode gives 0 and decode gives back the constant via offset).
        diff[diff == 0.0] = 1.0

        self._scale  = (diff / 255.0).astype(np.float32)
        self._offset = lo.astype(np.float32)
        self._fitted = True

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, vector: np.ndarray) -> np.ndarray:
        """float32 → uint8.  Shape: (dim,) → (dim,)."""
        self._check_fitted()
        v = vector.astype(np.float32)
        q = (v - self._offset) / self._scale
        return np.clip(np.round(q), 0, 255).astype(np.uint8)

    def decode(self, quantized: np.ndarray) -> np.ndarray:
        """uint8 → float32 (approximate reconstruction).  Shape preserved."""
        self._check_fitted()
        return quantized.astype(np.float32) * self._scale + self._offset

    def encode_batch(self, vectors: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """float32 matrix (n, dim) → uint8 matrix (n, dim)."""
        self._check_fitted()
        m = np.stack(vectors, axis=0).astype(np.float32) if not isinstance(vectors, np.ndarray) else vectors.astype(np.float32)
        q = (m - self._offset) / self._scale
        return np.clip(np.round(q), 0, 255).astype(np.uint8)

    def decode_batch(self, quantized: np.ndarray) -> np.ndarray:
        """uint8 matrix (n, dim) → float32 matrix (n, dim)."""
        self._check_fitted()
        return quantized.astype(np.float32) * self._scale + self._offset

    # ------------------------------------------------------------------
    # Asymmetric distance
    # ------------------------------------------------------------------

    def asymmetric_distance(
        self, query_f32: np.ndarray, stored_uint8: np.ndarray
    ) -> float:
        """
        Fast dot-product proxy: dot(query_float32, decode(stored_uint8)).

        Asymmetric: the query is NOT quantized (no loss on query side).
        The stored vector is decoded inline — avoids allocating a full
        float32 copy when only the dot product is needed.

        Returns the raw dot product (higher = more similar).
        Use 1.0 − result for cosine distance.
        """
        self._check_fitted()
        decoded = stored_uint8.astype(np.float32) * self._scale + self._offset
        return float(np.dot(query_f32, decoded))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Saves scale and offset to <path>.npz."""
        self._check_fitted()
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez(path, scale=self._scale, offset=self._offset)

    def load(self, path: str) -> None:
        """Loads scale and offset from <path>.npz."""
        data = np.load(path + ".npz")
        self._scale  = data["scale"].astype(np.float32)
        self._offset = data["offset"].astype(np.float32)
        self._fitted = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def compression_ratio(self) -> float:
        """Always 4.0 for SQ8 (float32 = 4 bytes, uint8 = 1 byte)."""
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
        return f"SQ8Quantizer(dim={dim}, compression=4x)"


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dim = 64

    # Simulate unit-normalized embeddings
    vecs = [rng.standard_normal(dim).astype(np.float32) for _ in range(500)]
    vecs = [v / np.linalg.norm(v) for v in vecs]

    q = SQ8Quantizer()
    q.fit(vecs)
    print(q)

    v = vecs[0]
    enc = q.encode(v)
    dec = q.decode(enc)

    reconstruction_error = float(np.linalg.norm(v - dec))
    cosine_before = float(np.dot(v, v))           # 1.0 (self)
    cosine_after  = float(np.dot(v, dec))         # ≈ 1.0
    asym_dot      = q.asymmetric_distance(v, enc)

    print(f"Reconstruction L2 error : {reconstruction_error:.6f}")
    print(f"Cosine (exact)          : {cosine_before:.6f}")
    print(f"Cosine (decoded)        : {cosine_after:.6f}")
    print(f"Asymmetric dot product  : {asym_dot:.6f}")
    print(f"Compression ratio       : {q.compression_ratio}x")
