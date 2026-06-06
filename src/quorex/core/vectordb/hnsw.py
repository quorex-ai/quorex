"""
quorex.core.vectordb.hnsw
--------------------------
Hierarchical Navigable Small World (HNSW) index.
True Approximate Nearest Neighbor search in O(log n).

Pure Python + numpy — no external dependencies.

Based on: Malkov & Yashunin, 2018
"""

from __future__ import annotations

import numpy as np
import math
import random
import heapq
from dataclasses import dataclass, field


@dataclass
class HNSWNode:
    id: int
    vector: np.ndarray | None          # float32, normalized; None when quantized
    neighbors: dict[int, list[int]] = field(default_factory=dict)
    quantized: np.ndarray | None = None  # uint8 — set when SQ8 is active
    # neighbors[layer] = [node_id, ...]


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
        quantizer=None,
    ):
        """
        dim             : vector dimensions
        M               : max connections per node per layer
        ef_construction : candidate list size during build (quality vs speed)
        ef_search       : candidate list size during search (recall vs latency)
                          lower ef_search = faster but less accurate (ANN)
                          higher ef_search = slower but more accurate (→ KNN)
        seed            : reproducibility — uses LOCAL random.Random instance
                          so different HNSWIndex instances don't perturb each
                          other's level sampling.
        """
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1 / math.log(M)

        self.nodes: dict[int, HNSWNode] = {}
        self.entry_point: int | None = None
        self.max_layer: int = 0

        # SQ8 quantizer — when set, vectors are stored as uint8 to save 4x RAM.
        # Read-only after fit: no lock needed for concurrent reads.
        self.quantizer = quantizer

        # Local RNG — does not affect global random / np.random state.
        # Why: Segment creates one HNSWIndex per user. With a global seed,
        # creating a new index would reset the global RNG and perturb all
        # other indexes' _random_level() calls.
        self._seed = seed
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, id: int, vector: np.ndarray) -> None:
        """Inserts a vector into the index.  Quantizes after wiring if SQ8 is active."""
        vector = self._normalize(vector)
        node = HNSWNode(id=id, vector=vector)
        level = self._random_level()
        self.nodes[id] = node

        # First node — initialize empty neighbor lists for all its layers.
        if self.entry_point is None:
            for l in range(level + 1):
                node.neighbors[l] = []
            self.entry_point = id
            self.max_layer = level
            return

        # Pre-initialize neighbor lists for every layer this node lives on.
        # Avoids the old bug where layers above max_layer were left
        # uninitialized when the new node extended the index height.
        for l in range(level + 1):
            node.neighbors.setdefault(l, [])

        ep = self.entry_point

        # Phase 1 — greedy descent above insertion level
        for l in range(self.max_layer, level, -1):
            ep = self._greedy_search(vector, ep, l)

        # Phase 2 — beam search + connect at each level
        for l in range(min(level, self.max_layer), -1, -1):
            candidates = self._search_layer_heap(vector, ep, self.ef_construction, l)
            max_conn = self.M if l > 0 else self.M_max0
            neighbors = candidates[:max_conn]

            node.neighbors[l] = [n_id for n_id, _ in neighbors]

            for n_id, _ in neighbors:
                n_node = self.nodes[n_id]
                if l not in n_node.neighbors:
                    n_node.neighbors[l] = []
                n_node.neighbors[l].append(id)

                if len(n_node.neighbors[l]) > max_conn:
                    n_node.neighbors[l] = self._prune(
                        self._get_float32(n_node), n_node.neighbors[l], max_conn
                    )

            if candidates:
                ep = candidates[0][0]

        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = id

        # Quantize after graph wiring — node.vector must be float32 throughout
        # the insertion so _prune can read it from neighboring nodes.
        if self.quantizer is not None:
            node.quantized = self.quantizer.encode(node.vector)
            node.vector = None

    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """
        ANN search — returns top_k nearest neighbors as (id, distance).
        Distance = cosine distance (lower = more similar).

        Two-phase when SQ8 is active:
          Phase 1 — ANN with asymmetric uint8 distance (fast, approximate)
          Phase 2 — exact rerank on decoded float32 (corrects quantization error)
        """
        if self.entry_point is None:
            return []

        query = self._normalize(query)
        ep = self.entry_point

        for l in range(self.max_layer, 0, -1):
            ep = self._greedy_search(query, ep, l)

        # Fetch more candidates in quantized mode so phase-2 rerank has room.
        ef = max(self.ef_search, top_k * 2 if self.quantizer else top_k)
        candidates = self._search_layer_heap(query, ep, ef, 0)

        if self.quantizer is None:
            return candidates[:top_k]

        # Phase 2 — exact rerank using decoded float32
        reranked = []
        for nid, _ in candidates:
            node = self.nodes.get(nid)
            if node is None:
                continue
            f32 = self._normalize(self._get_float32(node))
            reranked.append((nid, self._dist(query, f32)))
        reranked.sort(key=lambda x: x[1])
        return reranked[:top_k]

    def delete(self, id: int) -> bool:
        """
        Hard delete — removes the node and detaches it from all neighbors.

        Returns True if the node existed and was removed.

        Note: HNSW does not natively support deletion without recall loss.
        For accuracy-critical workloads, prefer rebuilding the index via
        rebuild_from(...) after a batch of deletes.
        """
        if id not in self.nodes:
            return False

        node = self.nodes.pop(id)

        for layer, neighbors in node.neighbors.items():
            for n_id in neighbors:
                n_node = self.nodes.get(n_id)
                if n_node is None:
                    continue
                lst = n_node.neighbors.get(layer)
                if lst and id in lst:
                    lst.remove(id)

        # If we removed the entry point, pick a new one from the highest
        # remaining layer. If the index is empty, reset entirely.
        if self.entry_point == id:
            if not self.nodes:
                self.entry_point = None
                self.max_layer = 0
            else:
                new_max = max(
                    (max(n.neighbors.keys()) if n.neighbors else 0)
                    for n in self.nodes.values()
                )
                self.max_layer = new_max
                for nid, n in self.nodes.items():
                    if new_max in n.neighbors:
                        self.entry_point = nid
                        break
                else:
                    self.entry_point = next(iter(self.nodes))

        return True

    def update(self, id: int, vector: np.ndarray) -> bool:
        """
        Updates a node's vector by deleting + re-inserting.

        Returns True if the node existed.

        Re-insert is required because changing a vector invalidates
        the existing graph neighborhoods (they were chosen against the
        old vector).
        """
        if id not in self.nodes:
            return False
        self.delete(id)
        self.insert(id, vector)
        return True

    def enable_quantization(self, quantizer) -> None:
        """
        Quantizes all existing float32 vectors in-place to uint8.
        Sets the quantizer so future inserts are also quantized.
        """
        self.quantizer = quantizer
        n = 0
        for node in self.nodes.values():
            if node.vector is not None:
                node.quantized = quantizer.encode(node.vector)
                node.vector = None
                n += 1
        print(f"Quantized {n} vectors — RAM reduced by 4x")

    def rebuild_from(self, ids_and_vectors: list[tuple[int, np.ndarray]]) -> None:
        """
        Clears the index and rebuilds from a list of (id, vector).

        Used by the segment layer for periodic compaction after many
        soft-deletes / updates have degraded graph quality.
        """
        self.nodes.clear()
        self.entry_point = None
        self.max_layer = 0
        self._rng = random.Random(self._seed)
        for id, vec in ids_and_vectors:
            self.insert(id, vec)

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(nodes={len(self)}, dim={self.dim}, "
            f"M={self.M}, ef_search={self.ef_search}, layers={self.max_layer + 1})"
        )

    # ------------------------------------------------------------------
    # Persistence — topology only. Vectors live in storage.vectors.bin.
    # ------------------------------------------------------------------

    def topology_dict(self) -> dict:
        """
        Returns the graph topology (no vectors) as a JSON-serializable dict.
        Lets storage.load() rebuild the graph in O(n) without re-inserting.
        """
        return {
            "dim": self.dim,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "seed": self._seed,
            "entry_point": self.entry_point,
            "max_layer": self.max_layer,
            "nodes": {
                str(nid): {str(layer): nb for layer, nb in node.neighbors.items()}
                for nid, node in self.nodes.items()
            },
        }

    @classmethod
    def from_topology(
        cls,
        topology: dict,
        vectors_by_id: dict[int, np.ndarray],
        quantizer=None,
    ) -> "HNSWIndex":
        """
        Reconstructs an HNSWIndex from a topology dict + a vector map.

        quantizer : if provided, vectors_by_id contains uint8 arrays and are
                    stored as quantized (node.vector = None, node.quantized set).
        """
        idx = cls(
            dim=topology["dim"],
            M=topology["M"],
            ef_construction=topology["ef_construction"],
            ef_search=topology["ef_search"],
            seed=topology["seed"],
            quantizer=quantizer,
        )
        idx.entry_point = topology["entry_point"]
        idx.max_layer = topology["max_layer"]
        for nid_str, layer_map in topology["nodes"].items():
            nid = int(nid_str)
            vec = vectors_by_id.get(nid)
            if vec is None:
                continue
            neighbors = {int(layer): list(nb) for layer, nb in layer_map.items()}
            if quantizer is not None:
                idx.nodes[nid] = HNSWNode(
                    id=nid,
                    vector=None,
                    neighbors=neighbors,
                    quantized=vec,  # already uint8
                )
            else:
                idx.nodes[nid] = HNSWNode(
                    id=nid,
                    vector=idx._normalize(vec),
                    neighbors=neighbors,
                )
        return idx

    # ------------------------------------------------------------------
    # Core — heapq-based ANN search
    # ------------------------------------------------------------------

    def _search_layer_heap(
        self, query: np.ndarray, ep_id: int, ef: int, layer: int
    ) -> list[tuple[int, float]]:
        """True ANN beam search using binary heaps (heapq)."""
        ep_dist = self._node_dist(query, self.nodes[ep_id])
        visited = {ep_id}

        candidates = [(ep_dist, ep_id)]
        heapq.heapify(candidates)

        results = [(-ep_dist, ep_id)]
        heapq.heapify(results)

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)

            worst_result_dist = -results[0][0]
            if c_dist > worst_result_dist:
                break

            for n_id in self.nodes[c_id].neighbors.get(layer, []):
                if n_id in visited:
                    continue
                visited.add(n_id)
                if n_id not in self.nodes:
                    continue
                n_dist = self._node_dist(query, self.nodes[n_id])
                worst = -results[0][0]

                if n_dist < worst or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, n_id))
                    heapq.heappush(results, (-n_dist, n_id))

                    if len(results) > ef:
                        heapq.heappop(results)

        out = [(-d, nid) for d, nid in results]
        out.sort(key=lambda x: x[0])
        return [(nid, dist) for dist, nid in out]

    def _greedy_search(self, query: np.ndarray, ep_id: int, layer: int) -> int:
        """Single-step greedy descent for upper layers."""
        current = ep_id
        current_dist = self._node_dist(query, self.nodes[current])

        improved = True
        while improved:
            improved = False
            for n_id in self.nodes[current].neighbors.get(layer, []):
                if n_id not in self.nodes:
                    continue
                d = self._node_dist(query, self.nodes[n_id])
                if d < current_dist:
                    current, current_dist = n_id, d
                    improved = True

        return current

    def _prune(self, vector: np.ndarray, neighbor_ids: list[int], max_conn: int) -> list[int]:
        """Keeps max_conn closest neighbors using a heap.
        `vector` is always float32 (the center node's vector during insert)."""
        heap = [
            (self._node_dist(vector, self.nodes[n]), n)
            for n in neighbor_ids
            if n in self.nodes
        ]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(min(max_conn, len(heap)))]

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random()) * self.ml)

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))

    def _node_dist(self, query: np.ndarray, node: "HNSWNode") -> float:
        """
        Cosine distance between a float32 query and a node.
        Uses asymmetric uint8 distance when quantizer is active (Phase 1).
        Falls back to exact float32 dot otherwise.
        """
        if self.quantizer is not None and node.quantized is not None:
            return 1.0 - self.quantizer.asymmetric_distance(query, node.quantized)
        return self._dist(query, node.vector)

    def _get_float32(self, node: "HNSWNode") -> np.ndarray:
        """Returns the float32 vector for a node, decoding from uint8 if needed."""
        if node.vector is not None:
            return node.vector
        return self.quantizer.decode(node.quantized)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v


if __name__ == "__main__":
    import time
    from quorex.core.embeddings.encoder import Encoder


    print("Building HNSW index (ANN)...\n")

    events = [
        {"action": "viewed_pricing", "metadata": {"plan": "pro", "source": "dashboard"}},
        {"action": "searched_pricing", "metadata": {"query": "pricing", "source": "web"}},
        {"action": "upgraded_plan", "metadata": {"plan": "pro", "source": "billing"}},
        {"action": "visited_homepage", "metadata": {"source": "organic"}},
        {"action": "clicked_cta", "metadata": {"source": "dashboard", "plan": "pro"}},
        {"action": "viewed_pricing", "metadata": {"plan": "starter", "source": "email"}},
        {"action": "opened_onboarding", "metadata": {"source": "welcome_email"}},
        {"action": "searched_docs", "metadata": {"query": "api reference", "source": "docs"}},
    ]

    encoder = Encoder(n_components=8)
    encoder.fit(events)

    index = HNSWIndex(dim=8, M=4, ef_construction=20, ef_search=10)
    for i, event in enumerate(events):
        index.insert(i, encoder.encode(event))

    print(index)

    query = {"action": "viewed_pricing", "metadata": {"plan": "pro"}}
    query_vec = encoder.encode(query)

    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        results = index.search(query_vec, top_k=3)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / N * 1000
    print(f"\nLatency: {avg_ms:.3f}ms avg over {N} queries")
    print("\nTop 3 results:")
    for node_id, dist in results:
        print(f"  id={node_id} similarity={round(1-dist, 4)} → {events[node_id]['action']}")

    # Hard delete + update demo
    print("\n--- Hard delete id=0 ---")
    index.delete(0)
    for node_id, dist in index.search(query_vec, top_k=3):
        print(f"  id={node_id} similarity={round(1-dist, 4)} → {events[node_id]['action']}")

    print("\n--- Update id=1 with new vector ---")
    new_vec = encoder.encode({"action": "totally_different", "metadata": {"x": "y"}})
    index.update(1, new_vec)
    print(f"  index now: {index}")
