from __future__ import annotations

import json
import os
import struct
import numpy as np
from .segment import Segment
from .hnsw import HNSWIndex


SNAPSHOT_VERSION = 2


class Storage:
    def __init__(self, path: str):
        """
        path: directory where data is stored.

        Files (version 2, float32):
        - vectors.bin     : float32 matrix of LIVE vectors only
        - meta.json       : { user_id: { vec_id: meta } }
        - index.json      : { version, dim, users: {...}, quantized: bool }
        - topology.json   : { user_id: HNSWIndex.topology_dict() }

        Additional files when SQ8 quantization is active:
        - vectors_sq8.bin : uint8 matrix (4x smaller than vectors.bin)
        - quantizer.npz   : scale + offset arrays for SQ8Quantizer
        """
        self.path = path
        self._vectors_path   = os.path.join(path, "vectors.bin")
        self._sq8_path       = os.path.join(path, "vectors_sq8.bin")
        self._quantizer_path = os.path.join(path, "quantizer")   # .npz added by np.savez
        self._meta_path      = os.path.join(path, "meta.json")
        self._index_path     = os.path.join(path, "index.json")
        self._topology_path  = os.path.join(path, "topology.json")

    # Save

    def save(self, segment: Segment, quantizer=None) -> None:
        """
        Persists the full segment to disk.

        quantizer : SQ8Quantizer instance — when provided, vectors are saved
                    as uint8 in vectors_sq8.bin instead of float32 vectors.bin.
        """
        os.makedirs(self.path, exist_ok=True)

        is_quantized = quantizer is not None and quantizer.is_fitted

        index = {
            "version":   SNAPSHOT_VERSION,
            "dim":       segment.dim,
            "quantized": is_quantized,
            "users":     {},
        }

        all_float32: list[np.ndarray] = []
        all_uint8:   list[np.ndarray] = []
        offset = 0
        topology = {}

        for user_id, hnsw in segment._indexes.items():
            live_ids = sorted(hnsw.nodes.keys())  # deterministic order

            for vec_id in live_ids:
                node = hnsw.nodes[vec_id]
                if is_quantized:
                    if node.quantized is not None:
                        all_uint8.append(node.quantized)
                    else:
                        # Node predates quantization — encode on the fly.
                        all_uint8.append(quantizer.encode(node.vector))
                else:
                    all_float32.append(node.vector.astype(np.float32))

            index["users"][user_id] = {
                "count":   len(live_ids),
                "offset":  offset,
                "ids":     live_ids,
                "counter": segment._counters.get(user_id, 0),
            }
            offset  += len(live_ids)
            topology[user_id] = hnsw.topology_dict()

        # Write vector data
        if is_quantized:
            if all_uint8:
                matrix = np.stack(all_uint8, axis=0).astype(np.uint8)
                with open(self._sq8_path, "wb") as f:
                    f.write(struct.pack(">II", len(all_uint8), segment.dim))
                    f.write(matrix.tobytes())
            elif os.path.exists(self._sq8_path):
                os.remove(self._sq8_path)
            quantizer.save(self._quantizer_path)
            # Remove stale float32 file if present
            if os.path.exists(self._vectors_path):
                os.remove(self._vectors_path)
        else:
            if all_float32:
                matrix = np.stack(all_float32, axis=0).astype(np.float32)
                with open(self._vectors_path, "wb") as f:
                    f.write(struct.pack(">II", len(all_float32), segment.dim))
                    f.write(matrix.tobytes())
            elif os.path.exists(self._vectors_path):
                os.remove(self._vectors_path)

        # Write meta.json — only live vec_ids
        serializable_meta = {
            user_id: {
                str(vec_id): meta
                for vec_id, meta in segment._metadata[user_id].items()
                if vec_id in segment._indexes[user_id].nodes
            }
            for user_id in segment._metadata
        }
        with open(self._meta_path, "w") as f:
            json.dump(serializable_meta, f, indent=2)

        with open(self._index_path, "w") as f:
            json.dump(index, f, indent=2)

        with open(self._topology_path, "w") as f:
            json.dump(topology, f)

        total = sum(u["count"] for u in index["users"].values())
        mode  = "SQ8/uint8" if is_quantized else "float32"
        print(f"Saved {total} vectors ({mode}) across {len(index['users'])} users → {self.path}/")

    # Load

    def load(self, segment: Segment):
        """
        Loads a persisted segment from disk.  Returns the SQ8Quantizer if the
        snapshot was saved in quantized mode, otherwise None.

        Backward compatible: snapshots without a "quantized" flag load as float32.
        """
        if not self._exists():
            raise FileNotFoundError(f"No storage found at {self.path}")

        with open(self._index_path, "r") as f:
            index = json.load(f)

        version = index.get("version", 1)
        if version > SNAPSHOT_VERSION:
            raise RuntimeError(
                f"Snapshot version {version} is newer than supported "
                f"({SNAPSHOT_VERSION}). Upgrade quorex."
            )

        is_quantized = index.get("quantized", False)

        with open(self._meta_path, "r") as f:
            raw_meta = json.load(f)

        topology = {}
        if os.path.exists(self._topology_path):
            with open(self._topology_path, "r") as f:
                topology = json.load(f)

        # Load quantizer if needed
        quantizer = None
        if is_quantized:
            from .quantizer import SQ8Quantizer
            quantizer = SQ8Quantizer()
            quantizer.load(self._quantizer_path)

        # Load vector matrix
        if is_quantized:
            with open(self._sq8_path, "rb") as f:
                n_vectors, dim = struct.unpack(">II", f.read(8))
                raw = f.read(n_vectors * dim)
                matrix = np.frombuffer(raw, dtype=np.uint8).reshape(n_vectors, dim)
        else:
            with open(self._vectors_path, "rb") as f:
                n_vectors, dim = struct.unpack(">II", f.read(8))
                raw = f.read(n_vectors * dim * 4)
                matrix = np.frombuffer(raw, dtype=np.float32).reshape(n_vectors, dim)

        users = index.get("users", {}) if version >= 2 else index

        for user_id, info in users.items():
            offset = info["offset"]
            count  = info["count"]
            ids    = info.get("ids", list(range(count)))  # v1 fallback
            user_meta_raw = raw_meta.get(user_id, {})
            user_meta = {int(k): v for k, v in user_meta_raw.items()}

            vectors_by_id = {
                vec_id: matrix[offset + i] for i, vec_id in enumerate(ids)
            }

            user_topology = topology.get(user_id)
            if user_topology and version >= 2:
                idx = HNSWIndex.from_topology(user_topology, vectors_by_id, quantizer=quantizer)
                segment._indexes[user_id] = idx
                segment._metadata[user_id] = {
                    vec_id: user_meta.get(vec_id, {}) for vec_id in ids
                }
                segment._counters[user_id] = info.get("counter", max(ids, default=-1) + 1)
                segment._pending_deletes[user_id] = 0
            else:
                # Legacy path — float32 only (no quantization on legacy snapshots).
                for vec_id in ids:
                    meta = user_meta.get(vec_id, {})
                    segment.insert_with_id(user_id, vec_id, vectors_by_id[vec_id], meta)

        mode = "SQ8/uint8" if is_quantized else "float32"
        print(f"Loaded {n_vectors} vectors ({mode}) across {len(users)} users ← {self.path}/")
        return quantizer

    # Utils

    def _exists(self) -> bool:
        vectors_exist = (
            os.path.exists(self._vectors_path)
            or os.path.exists(self._sq8_path)
        )
        return (
            vectors_exist
            and os.path.exists(self._meta_path)
            and os.path.exists(self._index_path)
        )

    def size_bytes(self) -> dict[str, int]:
        sizes = {}
        for name, path in [
            ("vectors.bin",     self._vectors_path),
            ("vectors_sq8.bin", self._sq8_path),
            ("quantizer.npz",   self._quantizer_path + ".npz"),
            ("meta.json",       self._meta_path),
            ("index.json",      self._index_path),
            ("topology.json",   self._topology_path),
        ]:
            sizes[name] = os.path.getsize(path) if os.path.exists(path) else 0
        return sizes

    def __repr__(self) -> str:
        sizes = self.size_bytes()
        total = sum(sizes.values())
        return f"Storage(path={self.path}, size={total}B)"


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

    segment = Segment(dim=n_dims, M=4, ef_construction=20, ef_search=10)
    segment.insert_batch("user_123", encoder.encode_batch(events_123), events_123)
    segment.insert_batch("user_456", encoder.encode_batch(events_456), events_456)

    # Delete one to test that deleted vecs don't end up on disk
    segment.delete("user_123", 0)
    print(segment)

    storage = Storage("/tmp/quorex_storage")
    storage.save(segment)
    print(storage)

    print("\n--- Reloading from disk ---")
    segment2 = Segment(dim=n_dims, M=4, ef_construction=20, ef_search=10)
    storage.load(segment2)
    print(segment2)
    print(f"user_123 live ids after reload: {segment2.live_ids('user_123')}")

    query_vec = encoder.encode({"action": "viewed_pricing", "metadata": {"plan": "pro"}})
    print("\nSearch user_123:")
    for r in segment2.search("user_123", query_vec, top_k=3):
        print(f"  score={r['score']} → {r['meta']['action']}")
