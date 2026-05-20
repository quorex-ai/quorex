import json
import os
import numpy as np
from encoder import Encoder
from similarity import SimilarityEngine


class VectorStore:
    def __init__(self, path: str):
        """
        path: directory where the store is saved.
        Example: "./data/vector_store"
        """
        self.path = path
        self.engine = SimilarityEngine()
        self.encoder: Encoder | None = None

        self._vectors_path = os.path.join(path, "vectors.npy")
        self._meta_path = os.path.join(path, "metadata.json")
        self._encoder_path = os.path.join(path, "encoder")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def init(self, encoder: Encoder) -> None:
        """
        Attaches a fitted encoder to the store.
        Must be called before add() or search().
        """
        if not encoder.fitted:
            raise RuntimeError("Encoder must be fitted before attaching to store.")
        self.encoder = encoder

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, event: dict, user_id: str) -> np.ndarray:
        """
        Encodes a single event and adds it to the store.
        Returns the embedding vector.
        """
        self._check_ready()

        vector = self.encoder.encode(event)
        meta = {"userId": user_id, **event}
        self.engine.add(vector, meta)

        return vector

    def add_batch(self, events: list[dict], user_id: str) -> np.ndarray:
        """
        Encodes and stores a batch of events for a given user.
        Returns the embedding matrix.
        """
        self._check_ready()

        vectors = self.encoder.encode_batch(events)
        metas = [{"userId": user_id, **e} for e in events]
        self.engine.add_batch(vectors, metas)

        return vectors

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_event: dict,
        user_id: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """
        Finds the most relevant past events for a given user + query.

        Filters results to the given user_id only.
        Returns list of { rank, score, meta }.
        """
        self._check_ready()

        query_vec = self.encoder.encode(query_event)
        all_results = self.engine.search(query_vec, top_k=len(self.engine), threshold=threshold)

        # Filter by user_id
        user_results = [r for r in all_results if r["meta"].get("userId") == user_id]

        # Re-rank and limit to top_k
        for i, r in enumerate(user_results[:top_k]):
            r["rank"] = i + 1

        return user_results[:top_k]

    def recall(self, user_id: str, query: str, top_k: int = 5) -> list[dict]:
        """
        High-level recall interface — takes a raw text query.
        Used by quorex.recall(userId, query).
        """
        self._check_ready()

        query_vec = self.encoder.encode_text(query)
        all_results = self.engine.search(query_vec, top_k=len(self.engine))

        user_results = [r for r in all_results if r["meta"].get("userId") == user_id]

        for i, r in enumerate(user_results[:top_k]):
            r["rank"] = i + 1

        return user_results[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Saves vectors, metadata and encoder to disk."""
        os.makedirs(self.path, exist_ok=True)

        # Save vectors
        if self.engine.index is not None:
            np.save(self._vectors_path, self.engine.index)

        # Save metadata
        with open(self._meta_path, "w") as f:
            json.dump(self.engine.metadata, f, indent=2)

        # Save encoder
        self.encoder.save(self._encoder_path)

        print(f"Store saved — {len(self.engine)} vectors → {self.path}/")

    def load(self) -> None:
        """Loads vectors, metadata and encoder from disk."""
        # Load encoder
        self.encoder = Encoder()
        self.encoder.load(self._encoder_path)

        # Load vectors
        vectors = np.load(self._vectors_path)

        # Load metadata
        with open(self._meta_path, "r") as f:
            metas = json.load(f)

        # Rebuild index
        self.engine.clear()
        self.engine.add_batch(vectors, metas)

        print(f"Store loaded — {len(self.engine)} vectors from {self.path}/")

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        if self.encoder is None:
            raise RuntimeError("Store has no encoder. Call init() or load() first.")

    def __len__(self) -> int:
        return len(self.engine)

    def __repr__(self) -> str:
        return f"VectorStore(path={self.path}, stored={len(self)})"


if __name__ == "__main__":
    events = [
        {"action": "viewed_pricing", "metadata": {"plan": "pro", "source": "dashboard"}},
        {"action": "searched_pricing", "metadata": {"query": "pricing", "source": "web"}},
        {"action": "upgraded_plan", "metadata": {"plan": "pro", "source": "billing"}},
        {"action": "visited_homepage", "metadata": {"source": "organic"}},
        {"action": "clicked_cta", "metadata": {"source": "dashboard", "plan": "pro"}},
        {"action": "viewed_pricing", "metadata": {"plan": "starter", "source": "email"}},
    ]

    # 1 — Fit encoder
    encoder = Encoder(n_components=4)
    encoder.fit(events)

    # 2 — Init store
    store = VectorStore("/tmp/quorex_store")
    store.init(encoder)

    # 3 — Add events for two users
    store.add_batch(events[:4], user_id="user_123")
    store.add_batch(events[4:], user_id="user_456")

    print(f"\n{store}\n")

    # 4 — Search
    query = {"action": "viewed_pricing", "metadata": {"plan": "pro"}}
    results = store.search(query, user_id="user_123", top_k=3)

    print("Search results for user_123:")
    for r in results:
        print(f"  #{r['rank']} score={r['score']} → {r['meta']['action']}")

    # 5 — Recall via raw text
    recalled = store.recall("user_123", "pricing pro", top_k=2)
    print("\nRecall 'pricing pro' for user_123:")
    for r in recalled:
        print(f"  #{r['rank']} score={r['score']} → {r['meta']['action']}")

    # 6 — Save + reload
    store.save()

    store2 = VectorStore("/tmp/quorex_store")
    store2.load()

    results2 = store2.search(query, user_id="user_123", top_k=3)
    print("\nAfter reload — same results:", [r["score"] for r in results] == [r["score"] for r in results2])