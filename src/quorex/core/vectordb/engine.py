from __future__ import annotations

import os
import threading
import time
import numpy as np
from .segment import Segment
from .wal import WAL, OpType
from .storage import Storage
from .consolidator import Consolidator, ConsolidationConfig


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
        bio_enabled: bool = True,
        consolidation_config: ConsolidationConfig | None = None,
    ):
        self.path = path
        self.dim = dim
        self.checkpoint_every = checkpoint_every
        self.compact_after_deletes = compact_after_deletes
        self.quantize = quantize
        self.bio_enabled = bio_enabled

        self._write_count = 0
        self._dirty = False
        self._lock = threading.RLock()

        self.segment = Segment(
            dim=dim,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            bio_enabled=bio_enabled,
        )
        self.wal     = WAL(os.path.join(path, "quorex.wal"))
        self.storage = Storage(os.path.join(path, "snapshot"))
        self.quantizer = None

        # Bio consolidator
        self.consolidator = Consolidator(consolidation_config)
        self._consolidation_config = consolidation_config or ConsolidationConfig()

    # Lifecycle

    def start(self) -> None:
        os.makedirs(self.path, exist_ok=True)

        try:
            loaded_quantizer = self.storage.load(self.segment)
            if loaded_quantizer is not None:
                self.quantizer = loaded_quantizer
                self.segment.quantizer = loaded_quantizer
            print("Snapshot loaded.")
        except FileNotFoundError:
            print("No snapshot found — starting fresh.")
        except Exception as e:
            # A corrupt or dimension-incompatible snapshot must never take the
            # whole engine down. Quarantine the bad files (the WAL belongs to the
            # same era) and start from an empty, healthy store instead of crashing.
            print(f"WARNING: snapshot load failed ({e!r}); quarantining and starting fresh.")
            self._quarantine_corrupt_state()
            self.wal.open()
            print(f"VectorDBEngine started (fresh after corrupt snapshot) → {self.path}/")
            return

        entries = self.wal.replay()
        cp = self.wal.last_checkpoint()
        to_replay = entries[cp + 1:] if cp >= 0 else entries

        if to_replay:
            print(f"Replaying {len(to_replay)} WAL entries...")
            for entry in to_replay:
                if entry.op == OpType.INSERT:
                    if entry.vector is None:
                        continue
                    vec = np.array(entry.vector, dtype=np.float32)
                    self.segment.insert_with_id(
                        entry.user_id, entry.vec_id, vec, entry.meta
                    )
                elif entry.op == OpType.UPDATE:
                    if entry.vector is None:
                        continue
                    vec = np.array(entry.vector, dtype=np.float32)
                    self.segment.update(
                        entry.user_id, entry.vec_id, vec, entry.meta
                    )
                elif entry.op == OpType.DELETE:
                    self.segment.delete(entry.user_id, entry.vec_id)

        self.wal.open()
        print(f"VectorDBEngine started → {self.path}/")

    def _quarantine_corrupt_state(self) -> None:
        """
        Move an unreadable snapshot + WAL aside (renamed *.corrupt-<ts>) and
        drop any partial in-memory state, so the engine can boot on an empty,
        healthy store instead of crash-looping on the same bad files.
        """
        stamp = int(time.time())
        for p in (self.storage.path, self.wal.path):
            try:
                if os.path.exists(p):
                    os.rename(p, f"{p}.corrupt-{stamp}")
                    print(f"  quarantined {p} → {p}.corrupt-{stamp}")
            except OSError as err:
                print(f"  WARNING: could not quarantine {p}: {err}")
        # Clear anything a partial load may have populated.
        for d in (self.segment._indexes, self.segment._metadata,
                  self.segment._counters, self.segment._pending_deletes):
            d.clear()
        self.quantizer = None
        self.segment.quantizer = None

    def stop(self) -> None:
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
        with self._lock:
            return [
                self.insert(user_id, vec, meta)
                for vec, meta in zip(vectors, metas)
            ]

    def update(
        self,
        user_id: str,
        vec_id: int,
        vector: np.ndarray,
        meta: dict | None = None,
    ) -> bool:
        with self._lock:
            existing_meta = (
                self.segment._metadata.get(user_id, {}).get(vec_id, {})
            )
            new_meta = meta if meta is not None else existing_meta
            self.wal.log_update(user_id, vec_id, new_meta, vector)
            ok = self.segment.update(user_id, vec_id, vector, new_meta)
            if ok:
                self._mark_write()
            return ok

    def delete(self, user_id: str, vec_id: int) -> bool:
        with self._lock:
            self.wal.log_delete(user_id, vec_id)
            ok = self.segment.delete(user_id, vec_id)
            if ok:
                self._mark_write()
                if (
                    self.segment.pending_deletes(user_id)
                    >= self.compact_after_deletes
                ):
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
        results = self.segment.search(user_id, query, top_k=top_k * 2)
        filtered = [r for r in results if r["score"] >= threshold]
        return filtered[:top_k]

    # MAINTENANCE

    def consolidate(self) -> dict:
        """
        Manual consolidation trigger.
        Also called automatically every consolidate_every inserts.
        """
        with self._lock:
            stats = self.consolidator.run(self.segment)
            if stats["pruned"] > 0 or stats["merged"] > 0:
                self._dirty = True
            print(
                f"Consolidation #{stats['cycle']} — "
                f"weights_updated={stats['weights_updated']} "
                f"pruned={stats['pruned']} "
                f"merged={stats['merged']} "
                f"({stats['duration_ms']:.1f}ms)"
            )
            return stats

    def enable_quantization(self) -> None:
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
            print("SQ8 quantization enabled — 4x RAM reduction active")

    def checkpoint(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self.storage.save(self.segment, quantizer=self.quantizer)
            self.wal.log_checkpoint()
            self.wal.truncate_after_checkpoint()
            self._dirty = False
            print(
                f"Checkpoint — {self.segment.total_vectors()} vectors saved."
            )

    def compact(self, user_id: str | None = None) -> int:
        with self._lock:
            n = self.segment.compact(user_id)
            self._mark_write()
            return n

    def _mark_write(self) -> None:
        self._dirty = True
        self._write_count += 1

        if (
            self.quantize
            and self.quantizer is None
            and self.segment.total_vectors() >= 100
        ):
            self.enable_quantization()

        cfg = self._consolidation_config
        if (
            self.bio_enabled
            and cfg.consolidate_every
            and self._write_count % cfg.consolidate_every == 0
        ):
            self.consolidate()

        if (
            self.checkpoint_every
            and self._write_count % self.checkpoint_every == 0
        ):
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
            "ram_reduction":       "4x" if self.quantizer else "1x",
            "quantizer_fitted":    (
                self.quantizer.is_fitted if self.quantizer else False
            ),
            "bio_enabled":         self.bio_enabled,
            "consolidation_cycle": self.consolidator._cycle_count,
        }

    def __repr__(self) -> str:
        return (
            f"VectorDBEngine(path={self.path}, "
            f"users={self.segment.user_count()}, "
            f"vectors={self.segment.total_vectors()}, "
            f"bio={'on' if self.bio_enabled else 'off'})"
        )