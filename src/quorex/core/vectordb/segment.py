from __future__ import annotations

import numpy as np
import time
from .hnsw import HNSWIndex


class Segment:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        bio_enabled: bool = True,
    ):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.bio_enabled = bio_enabled

        self._indexes: dict[str, HNSWIndex] = {}
        self._metadata: dict[str, dict[int, dict]] = {}
        self._counters: dict[str, int] = {}
        self._pending_deletes: dict[str, int] = {}
        self.quantizer = None

    # WRITE

    def insert(self, user_id: str, vector: np.ndarray, meta: dict) -> int:
        self._ensure_user(user_id)

        vec_id = self._counters[user_id]

        # Extract bio fields from meta if present
        inner = meta.get("metadata") or {}
        ts = inner.get("timestamp", time.time())
        reinforcements = inner.get("reinforcements", 1)

        self._indexes[user_id].insert(
            vec_id, vector,
            timestamp=ts,
            reinforcements=reinforcements,
        )
        self._metadata[user_id][vec_id] = meta
        self._counters[user_id] += 1

        return vec_id

    def insert_with_id(
        self, user_id: str, vec_id: int, vector: np.ndarray, meta: dict
    ) -> None:
        self._ensure_user(user_id)

        inner = meta.get("metadata") or {}
        ts = inner.get("timestamp", time.time())
        reinforcements = inner.get("reinforcements", 1)

        self._indexes[user_id].insert(
            vec_id, vector,
            timestamp=ts,
            reinforcements=reinforcements,
        )
        self._metadata[user_id][vec_id] = meta
        if vec_id >= self._counters[user_id]:
            self._counters[user_id] = vec_id + 1

    def insert_batch(
        self, user_id: str, vectors: np.ndarray, metas: list[dict]
    ) -> list[int]:
        return [
            self.insert(user_id, vec, meta)
            for vec, meta in zip(vectors, metas)
        ]

    def update(
        self, user_id: str, vec_id: int,
        vector: np.ndarray, meta: dict | None = None
    ) -> bool:
        if user_id not in self._indexes:
            return False
        ok = self._indexes[user_id].update(vec_id, vector)
        if not ok:
            return False
        if meta is not None:
            self._metadata[user_id][vec_id] = meta
        return True

    def delete(self, user_id: str, vec_id: int) -> bool:
        if user_id not in self._indexes:
            return False
        ok = self._indexes[user_id].delete(vec_id)
        if not ok:
            return False
        self._metadata[user_id].pop(vec_id, None)
        self._pending_deletes[user_id] = self._pending_deletes.get(user_id, 0) + 1
        return True

    def reinforce(self, user_id: str, vec_id: int) -> bool:
        """
        Increments reinforcement count on a node.
        Called by MemoryManager when a memory is recalled.
        """
        idx = self._indexes.get(user_id)
        if not idx:
            return False
        node = idx.nodes.get(vec_id)
        if node is None:
            return False
        node.reinforcements += 1
        node.timestamp = time.time()  # refresh timestamp on recall
        return True

    def set_conflict_score(
        self, user_id: str, vec_id: int, score: float
    ) -> bool:
        """Sets the conflict score on a node (0.0 = clean, 1.0 = conflicted)."""
        idx = self._indexes.get(user_id)
        if not idx:
            return False
        node = idx.nodes.get(vec_id)
        if node is None:
            return False
        node.conflict_score = max(0.0, min(1.0, score))
        return True

    # SEARCH

    def search(
        self, user_id: str, query: np.ndarray, top_k: int = 5
    ) -> list[dict]:
        if user_id not in self._indexes:
            return []

        results = self._indexes[user_id].search(query, top_k=top_k)
        return [
            {
                "id": vec_id,
                "score": round(1 - dist, 4),
                "meta": self._metadata[user_id].get(vec_id, {}),
            }
            for vec_id, dist in results
            if vec_id in self._metadata[user_id]
        ]

    # MAINTENANCE

    def compact(self, user_id: str | None = None) -> int:
        targets = [user_id] if user_id else list(self._indexes.keys())
        for uid in targets:
            idx = self._indexes.get(uid)
            if not idx:
                continue

            live = []
            bio_meta = {}

            for nid in idx.nodes:
                node = idx.nodes[nid]
                if node.vector is not None:
                    live.append((nid, node.vector))
                elif node.quantized is not None and idx.quantizer is not None:
                    live.append((nid, idx.quantizer.decode(node.quantized)))
                else:
                    continue

                bio_meta[nid] = {
                    "timestamp": node.timestamp,
                    "reinforcements": node.reinforcements,
                    "conflict_score": node.conflict_score,
                    "bio_weight": node.bio_weight,
                }

            idx.rebuild_from(live, bio_meta=bio_meta)
            self._pending_deletes[uid] = 0

        return len(targets)

    def pending_deletes(self, user_id: str) -> int:
        return self._pending_deletes.get(user_id, 0)

    # UTILS

    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self._indexes:
            self._indexes[user_id] = HNSWIndex(
                dim=self.dim,
                M=self.M,
                ef_construction=self.ef_construction,
                ef_search=self.ef_search,
                quantizer=self.quantizer,
                bio_enabled=self.bio_enabled,
            )
            self._metadata[user_id] = {}
            self._counters[user_id] = 0
            self._pending_deletes[user_id] = 0

    def user_count(self) -> int:
        return len(self._indexes)

    def vector_count(self, user_id: str) -> int:
        idx = self._indexes.get(user_id)
        return len(idx) if idx else 0

    def total_vectors(self) -> int:
        return sum(len(idx) for idx in self._indexes.values())

    def has_user(self, user_id: str) -> bool:
        return user_id in self._indexes

    def live_ids(self, user_id: str) -> list[int]:
        idx = self._indexes.get(user_id)
        return list(idx.nodes.keys()) if idx else []

    def get_vector(self, user_id: str, vec_id: int) -> np.ndarray | None:
        idx = self._indexes.get(user_id)
        if not idx:
            return None
        node = idx.nodes.get(vec_id)
        if node is None:
            return None
        if node.vector is not None:
            return node.vector
        if node.quantized is not None and idx.quantizer is not None:
            return idx.quantizer.decode(node.quantized)
        return None

    def __repr__(self) -> str:
        return (
            f"Segment(users={self.user_count()}, "
            f"total_vectors={self.total_vectors()}, "
            f"dim={self.dim})"
        )