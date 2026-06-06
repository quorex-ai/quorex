from __future__ import annotations

import os
import threading
import numpy as np
from .segment import Segment
from .wal import WAL, OpType
from .storage import Storage


class VectorDBEngine:
    def __init__(
        self,
        path: str,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        checkpoint_every: int = 100,
        compact_after_deletes: int = 1000,
        quantize: bool = False,
    ):
        """
        path                  : directory for all DB files
        dim                   : vector dimensions
        M                     : HNSW max connections per node
        ef_construction       : HNSW build quality
        ef_search             : HNSW search quality (lower = faster ANN)
        checkpoint_every      : auto-checkpoint every N writes (insert/update/delete)
        compact_after_deletes : auto-compact a user's graph after N deletes
        quantize              : auto-enable SQ8 after the first 100 vectors
        """
        self.path = path
        self.dim = dim
        self.checkpoint_every = checkpoint_every
        self.compact_after_deletes = compact_after_deletes
        self.quantize = quantize

        self._write_count = 0
        self._dirty = False  # true when state has changed since last checkpoint

        # Reentrant — checkpoint() can be called from inside insert() etc.
        self._lock = threading.RLock()

        self.segment = Segment(
            dim=dim,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        self.wal     = WAL(os.path.join(path, "quorex.wal"))
        self.storage = Storage(os.path.join(path, "snapshot"))

        # SQ8 quantizer — None until enable_quantization() is called.
        self.quantizer = None

    # Lifecycle

    def start(self) -> None:
        """
        Starts the engine.
        1. Load last snapshot if exists (restores quantizer state if present)
        2. Replay WAL entries after last checkpoint (with their vectors)
        """
        os.makedirs(self.path, exist_ok=True)

        try:
            loaded_quantizer = self.storage.load(self.segment)
            if loaded_quantizer is not None:
                self.quantizer = loaded_quantizer
                self.segment.quantizer = loaded_quantizer
            print("Snapshot loaded.")
        except FileNotFoundError:
            print("No snapshot found — starting fresh.")

        entries = self.wal.replay()
        cp = self.wal.last_checkpoint()
        to_replay = entries[cp + 1:] if cp >= 0 else entries

        if to_replay:
            print(f"Replaying {len(to_replay)} WAL entries...")
            for entry in to_replay:
                if entry.op == OpType.INSERT:
                    if entry.vector is None:
                        continue  # legacy WAL entry without vector
                    vec = np.array(entry.vector, dtype=np.float32)
                    self.segment.insert_with_id(
                        entry.user_id, entry.vec_id, vec, entry.meta
                    )
                elif entry.op == OpType.UPDATE:
                    if entry.vector is None:
                        continue
                    vec = np.array(entry.vector, dtype=np.float32)
                    self.segment.update(entry.user_id, entry.vec_id, vec, entry.meta)
                elif entry.op == OpType.DELETE:
                    self.segment.delete(entry.user_id, entry.vec_id)

        self.wal.open()
        print(f"VectorDBEngine started → {self.path}/")

    def stop(self) -> None:
        """Checkpoints and stops the engine."""
        with self._lock:
            if self._dirty:
                self.checkpoint()
            self.wal.close()
        print("VectorDBEngine stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # Write

    def insert(self, user_id: str, vector: np.ndarray, meta: dict) -> int:
        """
        Inserts a vector for a user.
        1. Log to WAL (with vector — full durability)
        2. Insert into HNSW segment
        """
        with self._lock:
            self.segment._ensure_user(user_id)
            vec_id = self.segment._counters[user_id]
            self.wal.log_insert(user_id, vec_id, meta, vector)
            assigned_id = self.segment.insert(user_id, vector, meta)
            self._mark_write()
            return assigned_id

    def insert_batch(
        self, user_id: str, vectors: np.ndarray, metas: list[dict]
    ) -> list[int]:
        """Inserts a batch of vectors for a user."""
        with self._lock:
            return [
                self.insert(user_id, vec, meta)
                for vec, meta in zip(vectors, metas)
            ]

    def update(
        self, user_id: str, vec_id: int, vector: np.ndarray, meta: dict | None = None
    ) -> bool:
        """
        Updates a vector. Returns True if it existed.
        WAL entry preserves the prior meta if a new one isn't supplied.
        """
        with self._lock:
            existing_meta = self.segment._metadata.get(user_id, {}).get(vec_id, {})
            new_meta = meta if meta is not None else existing_meta
            self.wal.log_update(user_id, vec_id, new_meta, vector)
            ok = self.segment.update(user_id, vec_id, vector, new_meta)
            if ok:
                self._mark_write()
            return ok

    def delete(self, user_id: str, vec_id: int) -> bool:
        """
        Hard delete — removed from HNSW graph + metadata.
        Returns True if the vector existed.
        """
        with self._lock:
            self.wal.log_delete(user_id, vec_id)
            ok = self.segment.delete(user_id, vec_id)
            if ok:
                self._mark_write()
                if self.segment.pending_deletes(user_id) >= self.compact_after_deletes:
                    self.compact(user_id)
            return ok

    # READ

    def search(
        self,
        user_id: str,
        query: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict]:
        """
        ANN search for a user. Filters by threshold.
        Read path doesn't need a lock — segment dicts are only mutated under
        the engine's write lock, and Python's GIL keeps individual dict
        reads atomic. For workloads with concurrent rebuilds though, callers
        should acquire the lock or run search in a snapshotting layer.
        """
        results = self.segment.search(user_id, query, top_k=top_k * 2)
        filtered = [r for r in results if r["score"] >= threshold]
        return filtered[:top_k]

    # MAINTENANCE

    def enable_quantization(self) -> None:
        """
        Fits SQ8 quantizer on current float32 vectors and quantizes in-place.
        Safe to call manually at any time after inserting enough vectors.
        No-op if already quantized.
        """
        with self._lock:
            if self.quantizer is not None:
                print("SQ8 already active.")
                return

            vectors = []
            for idx in self.segment._indexes.values():
                for node in idx.nodes.values():
                    if node.vector is not None:
                        vectors.append(node.vector)

            if not vectors:
                print("No float32 vectors to quantize.")
                return

            from .quantizer import SQ8Quantizer
            self.quantizer = SQ8Quantizer()
            self.quantizer.fit(vectors)
            self.segment.quantizer = self.quantizer

            for idx in self.segment._indexes.values():
                idx.enable_quantization(self.quantizer)

            self._dirty = True
            print(f"SQ8 quantization enabled — 4x RAM reduction active")

    def checkpoint(self) -> None:
        """
        Full snapshot + WAL checkpoint marker.
        Skips work if nothing changed since the last checkpoint.
        """
        with self._lock:
            if not self._dirty:
                return
            self.storage.save(self.segment, quantizer=self.quantizer)
            self.wal.log_checkpoint()
            self.wal.truncate_after_checkpoint()
            self._dirty = False
            print(f"Checkpoint — {self.segment.total_vectors()} vectors saved.")

    def compact(self, user_id: str | None = None) -> int:
        """Rebuilds HNSW graph(s) to reclaim quality after deletes/updates."""
        with self._lock:
            n = self.segment.compact(user_id)
            self._mark_write()
            return n

    def _mark_write(self) -> None:
        self._dirty = True
        self._write_count += 1
        # Auto-enable SQ8 after accumulating enough vectors to fit a good quantizer.
        if (
            self.quantize
            and self.quantizer is None
            and self.segment.total_vectors() >= 100
        ):
            self.enable_quantization()
        if self.checkpoint_every and self._write_count % self.checkpoint_every == 0:
            self.checkpoint()

    # STATS

    def stats(self) -> dict:
        return {
            "users":               self.segment.user_count(),
            "total_vectors":       self.segment.total_vectors(),
            "dim":                 self.dim,
            "wal_size_bytes":      self.wal.size_bytes(),
            "storage_size_bytes":  sum(self.storage.size_bytes().values()),
            "quantization":        "SQ8" if self.quantizer else "none",
            "ram_reduction":       "4x"  if self.quantizer else "1x",
            "quantizer_fitted":    self.quantizer.is_fitted if self.quantizer else False,
        }

    def __repr__(self) -> str:
        return (
            f"VectorDBEngine(path={self.path}, "
            f"users={self.segment.user_count()}, "
            f"vectors={self.segment.total_vectors()}, "
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
    ]
    all_events = events_123 + events_456

    n_dims = 5
    encoder = Encoder(n_components=n_dims)
    encoder.fit(all_events)

    print("=== Starting engine ===")
    with VectorDBEngine(
        path="/tmp/quorex_engine",
        dim=n_dims,
        M=4,
        ef_construction=20,
        ef_search=10,
        checkpoint_every=10,
    ) as engine:

        engine.insert_batch("user_123", encoder.encode_batch(events_123), events_123)
        engine.insert_batch("user_456", encoder.encode_batch(events_456), events_456)

        print(f"\n{engine}")
        print(f"Stats: {engine.stats()}")

        query_vec = encoder.encode({"action": "viewed_pricing", "metadata": {"plan": "pro"}})
        results = engine.search("user_123", query_vec, top_k=3)

        print("\nSearch results (user_123):")
        for r in results:
            print(f"  score={r['score']} → {r['meta']['action']}")

        engine.delete("user_123", 0)
        results_after = engine.search("user_123", query_vec, top_k=3)
        print("\nAfter HARD delete of vec_id=0:")
        for r in results_after:
            print(f"  score={r['score']} → {r['meta']['action']}")

    print("\n=== Restarting engine (recovery test) ===")
    with VectorDBEngine(
        path="/tmp/quorex_engine",
        dim=n_dims,
        M=4,
        ef_construction=20,
        ef_search=10,
    ) as engine2:
        print(engine2)
        results2 = engine2.search("user_123", query_vec, top_k=3)
        print("\nSearch after restart:")
        for r in results2:
            print(f"  score={r['score']} → {r['meta']['action']}")
