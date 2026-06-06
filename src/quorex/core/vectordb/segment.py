from __future__ import annotations

import numpy as np
from .hnsw import HNSWIndex


class Segment:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # One HNSW index per userId
        self._indexes: dict[str, HNSWIndex] = {}

        # Metadata store per userId
        self._metadata: dict[str, dict[int, dict]] = {}

        # Auto-increment id per userId — monotonically increasing, never
        # reused. Survives deletes so vec_ids remain stable references.
        self._counters: dict[str, int] = {}

        # Pending-deletes counter per user. The engine uses this to decide
        # when to trigger compaction (rebuild the HNSW graph).
        self._pending_deletes: dict[str, int] = {}

        # Quantizer — when set, new user indexes inherit it automatically.
        self.quantizer = None

    # WRITE

    def insert(self, user_id: str, vector: np.ndarray, meta: dict) -> int:
        """Inserts a vector for a given user. Returns assigned vec_id."""
        self._ensure_user(user_id)

        vec_id = self._counters[user_id]
        self._indexes[user_id].insert(vec_id, vector)
        self._metadata[user_id][vec_id] = meta
        self._counters[user_id] += 1

        return vec_id

    def insert_with_id(
        self, user_id: str, vec_id: int, vector: np.ndarray, meta: dict
    ) -> None:
        """
        Inserts a vector with an explicit vec_id (no auto-increment).
        Used by WAL replay and snapshot reload to preserve id continuity.
        Advances the counter so future inserts don't collide.
        """
        self._ensure_user(user_id)
        self._indexes[user_id].insert(vec_id, vector)
        self._metadata[user_id][vec_id] = meta
        if vec_id >= self._counters[user_id]:
            self._counters[user_id] = vec_id + 1

    def insert_batch(
        self, user_id: str, vectors: np.ndarray, metas: list[dict]
    ) -> list[int]:
        """Inserts a batch of vectors for a given user."""
        return [
            self.insert(user_id, vec, meta)
            for vec, meta in zip(vectors, metas)
        ]

    def update(self, user_id: str, vec_id: int, vector: np.ndarray, meta: dict | None = None) -> bool:
        """
        Updates a vector (and optionally its meta).
        Returns True if the (user, vec_id) pair existed.
        """
        if user_id not in self._indexes:
            return False
        ok = self._indexes[user_id].update(vec_id, vector)
        if not ok:
            return False
        if meta is not None:
            self._metadata[user_id][vec_id] = meta
        return True

    def delete(self, user_id: str, vec_id: int) -> bool:
        """
        Hard delete — removes the vector from the HNSW graph AND metadata.
        Returns True if it existed.
        """
        if user_id not in self._indexes:
            return False
        ok = self._indexes[user_id].delete(vec_id)
        if not ok:
            return False
        self._metadata[user_id].pop(vec_id, None)
        self._pending_deletes[user_id] = self._pending_deletes.get(user_id, 0) + 1
        return True

    # SEARCH

    def search(
        self, user_id: str, query: np.ndarray, top_k: int = 5
    ) -> list[dict]:
        """ANN search scoped to a single user's index."""
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
            if vec_id in self._metadata[user_id]  # safety: skip dangling ids
        ]

    # MAINTENANCE

    def compact(self, user_id: str | None = None) -> int:
        """
        Rebuilds the HNSW graph(s) from current live nodes. Recovers
        graph quality after many deletes/updates.

        Returns the number of users compacted.
        """
        targets = [user_id] if user_id else list(self._indexes.keys())
        for uid in targets:
            idx = self._indexes.get(uid)
            if not idx:
                continue
            live = []
            for nid in idx.nodes:
                node = idx.nodes[nid]
                if node.vector is not None:
                    live.append((nid, node.vector))
                elif node.quantized is not None and idx.quantizer is not None:
                    # Decode to float32; insert() will re-quantize.
                    live.append((nid, idx.quantizer.decode(node.quantized)))
            idx.rebuild_from(live)
            self._pending_deletes[uid] = 0
        return len(targets)

    def pending_deletes(self, user_id: str) -> int:
        return self._pending_deletes.get(user_id, 0)

    # UTILS

    def _ensure_user(self, user_id: str) -> None:
        """Creates a fresh HNSW index + metadata store for a new user."""
        if user_id not in self._indexes:
            self._indexes[user_id] = HNSWIndex(
                dim=self.dim,
                M=self.M,
                ef_construction=self.ef_construction,
                ef_search=self.ef_search,
                quantizer=self.quantizer,
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


if __name__ == "__main__":
    from quorex.core.embeddings.encoder import Encoder

    events_123 = [
        {"action": "viewed_pricing", "metadata": {"plan": "pro", "source": "dashboard"}},
        {"action": "upgraded_plan", "metadata": {"plan": "pro", "source": "billing"}},
        {"action": "clicked_cta", "metadata": {"source": "dashboard", "plan": "pro"}},
    ]
    events_456 = [
        {"action": "visited_homepage", "metadata": {"source": "organic"}},
        {"action": "searched_docs", "metadata": {"query": "api reference"}},
        {"action": "opened_onboarding", "metadata": {"source": "welcome_email"}},
    ]

    encoder = Encoder(n_components=8)
    encoder.fit(events_123 + events_456)

    segment = Segment(dim=8, M=4, ef_construction=20, ef_search=10)
    segment.insert_batch("user_123", encoder.encode_batch(events_123), events_123)
    segment.insert_batch("user_456", encoder.encode_batch(events_456), events_456)

    print(segment)

    query_vec = encoder.encode({"action": "viewed_pricing", "metadata": {"plan": "pro"}})
    print("\n--- Search user_123 ---")
    for r in segment.search("user_123", query_vec, top_k=3):
        print(f"  score={r['score']} → {r['meta']['action']}")

    print("\n--- Delete vec 0 + compact ---")
    segment.delete("user_123", 0)
    segment.compact("user_123")
    for r in segment.search("user_123", query_vec, top_k=3):
        print(f"  score={r['score']} → {r['meta']['action']}")

    print("\n--- Update vec 1 ---")
    new_vec = encoder.encode({"action": "viewed_pricing", "metadata": {"plan": "starter"}})
    segment.update("user_123", 1, new_vec, {"action": "viewed_pricing_starter"})
    for r in segment.search("user_123", query_vec, top_k=3):
        print(f"  score={r['score']} → {r['meta']['action']}")
