"""
quorex.core.vectordb.hnsw
--------------------------
Bio-HNSW — HNSW index with:
  - __slots__ on HNSWNode       (-100 bytes/node overhead, ~1MB @10K)
  - Factored asymmetric distance (precompute q_scaled/q_offset once per query)
  - Batched beam search (one matmul per node-expansion)
  - Quantization-aware construction (Piste 1)
  - Residual error compensation (Piste 2)
  - Bio-inspired periodic consolidation (bio_weight modulates distance)

Note: neighbors stay as dict[int, list[int]] — the int32 arena
optimization is deferred to the Rust port where it can be done
correctly with pre-allocated contiguous memory.
"""

from __future__ import annotations

import math
import heapq
import random
import time

import numpy as np


class HNSWNode:
    """
    HNSW node with __slots__ — no __dict__, no per-object overhead.
    Saves ~100 bytes/node vs a regular Python object (~1MB @10K nodes).
    """
    __slots__ = (
        "id",
        "vector",
        "quantized",
        "residual",
        "neighbors",        # dict[int, list[int]] — layer → neighbor ids
        "timestamp",
        "reinforcements",
        "conflict_score",
        "bio_weight",
    )

    def __init__(
        self,
        id: int,
        vector,
        timestamp: float = 0.0,
        reinforcements: int = 1,
        bio_weight: float = 1.0,
    ):
        self.id = id
        self.vector = vector
        self.quantized = None
        self.residual = None
        self.neighbors: dict[int, list[int]] = {}
        self.timestamp = timestamp
        self.reinforcements = reinforcements
        self.conflict_score = 0.0
        self.bio_weight = bio_weight


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
        quantizer=None,
        bio_enabled: bool = True,
    ):
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1 / math.log(M)
        self.bio_enabled = bio_enabled

        self.nodes: dict[int, HNSWNode] = {}
        self.entry_point: int | None = None
        self.max_layer: int = 0

        self.quantizer = quantizer
        self._seed = seed
        self._rng = random.Random(seed)

        # Per-query factored terms — set in search()/insert(), reset after.
        self._q_scaled = None
        self._q_offset: float = 0.0
        self._q_proj = None
        self._q_f32 = None

    # ── Public API ────────────────────────────────────────────────────

    def insert(
        self,
        id: int,
        vector: np.ndarray,
        timestamp: float | None = None,
        reinforcements: int = 1,
    ) -> None:
        vector = self._normalize(vector)
        node = HNSWNode(
            id=id,
            vector=vector,
            timestamp=timestamp or time.time(),
            reinforcements=reinforcements,
        )
        level = self._random_level()
        self.nodes[id] = node

        # Piste 1 — quantize BEFORE wiring so the graph topology is built
        # on the int8 distances used at search time (not float32).
        if self.quantizer is not None:
            node.quantized = self.quantizer.encode(vector)
            node.residual  = self.quantizer.encode_residual(vector)
            node.vector    = None
            self._set_query(vector)

        if self.entry_point is None:
            for l in range(level + 1):
                node.neighbors[l] = []
            self.entry_point = id
            self.max_layer   = level
            self._clear_query()
            return

        for l in range(level + 1):
            node.neighbors.setdefault(l, [])

        ep = self.entry_point

        # Phase 1 — greedy descent above insertion level
        for l in range(self.max_layer, level, -1):
            ep = self._greedy_search(vector, ep, l)

        # Phase 2 — beam search + connect at each level
        for l in range(min(level, self.max_layer), -1, -1):
            candidates = self._search_layer_heap(
                vector, ep, self.ef_construction, l
            )
            max_conn  = self.M if l > 0 else self.M_max0
            neighbors = candidates[:max_conn]

            node.neighbors[l] = [n_id for n_id, _ in neighbors]

            for n_id, _ in neighbors:
                n_node = self.nodes[n_id]
                if l not in n_node.neighbors:
                    n_node.neighbors[l] = []
                n_node.neighbors[l].append(id)

                if len(n_node.neighbors[l]) > max_conn:
                    n_node.neighbors[l] = self._prune(
                        self._get_float32(n_node),
                        n_node.neighbors[l],
                        max_conn,
                    )

            if candidates:
                ep = candidates[0][0]

        if level > self.max_layer:
            self.max_layer   = level
            self.entry_point = id

        self._clear_query()

    def search(
        self, query: np.ndarray, top_k: int = 5
    ) -> list[tuple[int, float]]:
        if self.entry_point is None:
            return []

        query = self._normalize(query)
        self._set_query(query)

        ep = self.entry_point
        for l in range(self.max_layer, 0, -1):
            ep = self._greedy_search(query, ep, l)

        ef = max(self.ef_search, top_k * 4 if self.quantizer else top_k)
        candidates = self._search_layer_heap(query, ep, ef, 0)

        if self.quantizer is None:
            self._clear_query()
            return candidates[:top_k]

        # Phase 2 — rerank using compensated distance (Piste 2).
        cand_ids = [nid for nid, _ in candidates if nid in self.nodes]
        if not cand_ids:
            self._clear_query()
            return []

        dists = self._batch_node_dist(query, cand_ids)
        order = np.argsort(dists)[:top_k]
        out   = [(cand_ids[i], float(dists[i])) for i in order]

        self._clear_query()
        return out

    def delete(self, id: int) -> bool:
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

        if self.entry_point == id:
            if not self.nodes:
                self.entry_point = None
                self.max_layer   = 0
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
        if id not in self.nodes:
            return False
        old = self.nodes[id]
        ts  = old.timestamp
        r   = old.reinforcements
        self.delete(id)
        self.insert(id, vector, timestamp=ts, reinforcements=r)
        return True

    def enable_quantization(self, quantizer) -> None:
        """Quantizes all existing float32 vectors + encodes residuals."""
        self.quantizer = quantizer
        n = 0
        for node in self.nodes.values():
            if node.vector is not None:
                node.residual  = quantizer.encode_residual(node.vector)
                node.quantized = quantizer.encode(node.vector)
                node.vector    = None
                n += 1
        print(
            f"Quantized {n} vectors with residual compensation "
            f"(R={quantizer.n_residual})"
        )

    def rebuild_from(
        self,
        ids_and_vectors: list[tuple[int, np.ndarray]],
        bio_meta: dict[int, dict] | None = None,
    ) -> None:
        self.nodes.clear()
        self.entry_point = None
        self.max_layer   = 0
        self._rng        = random.Random(self._seed)

        for id, vec in ids_and_vectors:
            meta = bio_meta.get(id, {}) if bio_meta else {}
            self.insert(
                id, vec,
                timestamp=meta.get("timestamp", time.time()),
                reinforcements=meta.get("reinforcements", 1),
            )
            if bio_meta and id in bio_meta:
                node = self.nodes.get(id)
                if node:
                    node.conflict_score = meta.get("conflict_score", 0.0)
                    node.bio_weight     = meta.get("bio_weight", 1.0)

    def apply_bio_weights(self, weights: dict[int, float]) -> None:
        for vec_id, w in weights.items():
            node = self.nodes.get(vec_id)
            if node is not None:
                node.bio_weight = max(0.0, min(1.0, w))

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(nodes={len(self)}, dim={self.dim}, "
            f"M={self.M}, ef_search={self.ef_search}, "
            f"bio={'on' if self.bio_enabled else 'off'}, "
            f"layers={self.max_layer + 1})"
        )

    # ── Persistence ───────────────────────────────────────────────────

    def topology_dict(self) -> dict:
        return {
            "dim":            self.dim,
            "M":              self.M,
            "ef_construction": self.ef_construction,
            "ef_search":      self.ef_search,
            "seed":           self._seed,
            "entry_point":    self.entry_point,
            "max_layer":      self.max_layer,
            "bio_enabled":    self.bio_enabled,
            "nodes": {
                str(nid): {
                    "neighbors": {
                        str(layer): nb
                        for layer, nb in node.neighbors.items()
                    },
                    "timestamp":      node.timestamp,
                    "reinforcements": node.reinforcements,
                    "conflict_score": node.conflict_score,
                    "bio_weight":     node.bio_weight,
                }
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
        idx = cls(
            dim=topology["dim"],
            M=topology["M"],
            ef_construction=topology["ef_construction"],
            ef_search=topology["ef_search"],
            seed=topology["seed"],
            quantizer=quantizer,
            bio_enabled=topology.get("bio_enabled", True),
        )
        idx.entry_point = topology["entry_point"]
        idx.max_layer   = topology["max_layer"]

        for nid_str, node_data in topology["nodes"].items():
            nid = int(nid_str)
            vec = vectors_by_id.get(nid)
            if vec is None:
                continue

            if isinstance(node_data, dict) and "neighbors" in node_data:
                layer_map      = node_data["neighbors"]
                ts             = node_data.get("timestamp", 0.0)
                reinforcements = node_data.get("reinforcements", 1)
                conflict_score = node_data.get("conflict_score", 0.0)
                bio_weight     = node_data.get("bio_weight", 1.0)
            else:
                layer_map      = node_data
                ts             = 0.0
                reinforcements = 1
                conflict_score = 0.0
                bio_weight     = 1.0

            neighbors = {
                int(layer): list(nb)
                for layer, nb in layer_map.items()
            }

            node = HNSWNode(
                id=nid,
                vector=None if quantizer is not None else idx._normalize(vec),
                timestamp=ts,
                reinforcements=reinforcements,
                bio_weight=bio_weight,
            )
            node.conflict_score = conflict_score
            node.neighbors      = neighbors

            if quantizer is not None:
                decoded       = quantizer.decode(vec) if vec.dtype == np.uint8 else vec
                node.quantized = vec
                node.residual  = quantizer.encode_residual(decoded)

            idx.nodes[nid] = node

        return idx

    # ── Query setup ───────────────────────────────────────────────────

    def _set_query(self, query_f32: np.ndarray) -> None:
        """Precompute factored terms ONCE per query / insert."""
        self._q_f32 = query_f32
        if self.quantizer is not None:
            self._q_scaled, self._q_offset = \
                self.quantizer.prepare_query(query_f32)
            self._q_proj = self.quantizer.project_query(query_f32)
        else:
            self._q_scaled = None
            self._q_offset = 0.0
            self._q_proj   = None

    def _clear_query(self) -> None:
        self._q_scaled = None
        self._q_offset = 0.0
        self._q_proj   = None
        self._q_f32    = None

    # ── Core search ───────────────────────────────────────────────────

    def _search_layer_heap(
        self, query: np.ndarray, ep_id: int, ef: int, layer: int
    ) -> list[tuple[int, float]]:
        ep_dist = self._node_dist(query, self.nodes[ep_id])
        visited = {ep_id}

        candidates = [(ep_dist, ep_id)]
        heapq.heapify(candidates)
        results = [(-ep_dist, ep_id)]
        heapq.heapify(results)

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            if c_dist > -results[0][0]:
                break

            neigh = [
                n_id for n_id in self.nodes[c_id].neighbors.get(layer, [])
                if n_id not in visited and n_id in self.nodes
            ]
            if not neigh:
                continue
            visited.update(neigh)

            dists = self._batch_node_dist(query, neigh)

            for n_id, nd in zip(neigh, dists):
                n_dist = float(nd)
                if n_dist < -results[0][0] or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, n_id))
                    heapq.heappush(results,    (-n_dist, n_id))
                    if len(results) > ef:
                        heapq.heappop(results)

        out = [(-d, nid) for d, nid in results]
        out.sort(key=lambda x: x[0])
        return [(nid, dist) for dist, nid in out]

    def _batch_node_dist(
        self, query: np.ndarray, node_ids: list[int]
    ) -> np.ndarray:
        """
        Distances for many nodes in one matmul.
        Quantized path uses the factored form — no per-node decode.
        """
        if self.quantizer is not None and self._q_scaled is not None:
            U = np.stack([self.nodes[n].quantized for n in node_ids])
            d = self.quantizer.factored_distance_batch(
                U, self._q_scaled, self._q_offset
            )
            # Residual compensation in batch (Piste 2)
            if self._q_proj is not None:
                residuals = [self.nodes[n].residual for n in node_ids]
                if all(r is not None for r in residuals):
                    R = np.stack(residuals).astype(np.float32)
                    d = d - (R @ self._q_proj)
        else:
            V = np.stack([self.nodes[n].vector for n in node_ids])
            d = 1.0 - (V @ query)

        if self.bio_enabled:
            w = np.fromiter(
                (self.nodes[n].bio_weight for n in node_ids),
                dtype=np.float32,
                count=len(node_ids),
            )
            if np.any(w < 1.0):
                d = d * (2.0 - w)

        return d

    def _greedy_search(
        self, query: np.ndarray, ep_id: int, layer: int
    ) -> int:
        current      = ep_id
        current_dist = self._node_dist(query, self.nodes[current])

        improved = True
        while improved:
            improved = False
            neigh = [
                n_id for n_id in self.nodes[current].neighbors.get(layer, [])
                if n_id in self.nodes
            ]
            if not neigh:
                break
            dists    = self._batch_node_dist(query, neigh)
            best_idx = int(np.argmin(dists))
            if float(dists[best_idx]) < current_dist:
                current      = neigh[best_idx]
                current_dist = float(dists[best_idx])
                improved     = True

        return current

    def _prune(
        self, vector: np.ndarray, neighbor_ids: list[int], max_conn: int
    ) -> list[int]:
        ids = [n for n in neighbor_ids if n in self.nodes]
        if not ids:
            return []
        dists = self._batch_node_dist(vector, ids)
        order = np.argsort(dists)[:max_conn]
        return [ids[i] for i in order]

    # ── Utils ─────────────────────────────────────────────────────────

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random()) * self.ml)

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))

    def _node_dist(self, query: np.ndarray, node: HNSWNode) -> float:
        """Single-node distance — used for entry point only."""
        if self.quantizer is not None and node.quantized is not None:
            if self._q_scaled is not None:
                dot = float(node.quantized @ self._q_scaled) + self._q_offset
                if node.residual is not None and self._q_proj is not None:
                    dot += float(
                        node.residual.astype(np.float32) @ self._q_proj
                    )
                cosine_d = 1.0 - dot
            else:
                cosine_d = 1.0 - self.quantizer.asymmetric_distance(
                    query, node.quantized
                )
        else:
            cosine_d = self._dist(query, node.vector)

        if self.bio_enabled and node.bio_weight < 1.0:
            return cosine_d * (2.0 - node.bio_weight)

        return cosine_d

    def _get_float32(self, node: HNSWNode) -> np.ndarray:
        if node.vector is not None:
            return node.vector
        return self.quantizer.decode(node.quantized)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v