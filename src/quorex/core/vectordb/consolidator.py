"""
quorex.core.vectordb.consolidator
-----------------------------------
Periodic bio-consolidation for Bio-HNSW.

Inspired by hippocampal consolidation — the brain's mechanism for
transferring episodic memories to long-term semantic storage during sleep.

Consolidation cycle (triggered every N inserts or on demand):

  1. Decay    — recompute bio_weight for every node using Ebbinghaus curve
  2. Reinforce — apply freq_boost from reinforcement count
  3. Conflict  — apply conflict_penalty from conflict_score
  4. Prune    — mark nodes below prune_threshold for deletion
  5. Merge    — fuse episodic nodes with cosine_sim > merge_threshold
                into a single semantic node (centroid)
  6. Apply    — push new bio_weights to HNSWIndex via apply_bio_weights()

Key change vs previous version:
  - Decay and freq_boost are now fully independent (no shared log term)
  - Scores are normalized relative to the best node in each user index
    so the cap at 1.0 never flattens the signal between nodes
  - freq_boost is linear (not log) on reinforcements so that a node
    with reinforcements=3 clearly outweighs reinforcements=1
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class ConsolidationConfig:
    # Decay — how fast memories fade with time.
    # Higher = slower decay (more stable memories).
    # decay = exp(-hours_ago / (24 * decay_stability_factor))
    decay_stability_factor: float = 2.0

    # Freq boost — how much reinforcement count amplifies the score.
    # freq_boost = 1.0 + freq_boost_factor * reinforcements
    # Linear (not log) so that r=3 clearly beats r=1.
    freq_boost_factor: float = 0.5

    # Conflict penalty — how much conflict degrades the score.
    conflict_penalty_factor: float = 0.4

    # Prune threshold — relative weight below which a node is pruned.
    # 0.05 means a node faded to 5% of the best node's score is removed.
    prune_threshold: float = 0.05

    # Merge threshold — cosine similarity above which two episodic nodes
    # are fused into a single semantic node.
    merge_threshold: float = 0.96

    max_merges_per_cycle: int = 50
    consolidate_every: int = 500


class Consolidator:
    """
    Stateless consolidation engine — operates on Segment + HNSWIndex objects.
    Called by VectorDBEngine._mark_write() when the write counter hits
    consolidate_every, or manually via engine.consolidator.run(segment).
    """

    def __init__(self, config: ConsolidationConfig | None = None):
        self.config = config or ConsolidationConfig()
        self._cycle_count = 0

    # ── Main entry point ──────────────────────────────────────────────

    def run(self, segment, engine_lock=None) -> dict:
        """
        Full consolidation cycle across all users in the segment.

        Returns a stats dict:
          { cycle, users, weights_updated, pruned, merged, duration_ms }
        """
        t0 = time.perf_counter()
        self._cycle_count += 1

        stats = {
            "cycle":           self._cycle_count,
            "users":           0,
            "weights_updated": 0,
            "pruned":          0,
            "merged":          0,
            "duration_ms":     0.0,
        }

        now = time.time()

        for user_id in list(segment._indexes.keys()):
            idx = segment._indexes.get(user_id)
            if not idx or not idx.nodes:
                continue

            stats["users"] += 1

            # Step 1 — Compute new bio_weights (normalized per user)
            new_weights, to_prune = self._compute_weights(idx, now)
            stats["weights_updated"] += len(new_weights)

            # Step 2 — Apply weights to index (O(n), no graph change)
            idx.apply_bio_weights(new_weights)

            # Step 3 — Prune faded nodes
            for vec_id in to_prune:
                segment.delete(user_id, vec_id)
            stats["pruned"] += len(to_prune)

            # Step 4 — Merge similar episodic nodes
            merged = self._merge_episodic(segment, user_id, idx, now)
            stats["merged"] += merged

        stats["duration_ms"] = (time.perf_counter() - t0) * 1000
        return stats

    # ── Weight computation ────────────────────────────────────────────

    def _compute_weights(
        self, idx, now: float
    ) -> tuple[dict[int, float], list[int]]:
        """
        Computes bio_weight for every node in one user's index.

        Key design decisions:
        1. Decay and freq_boost are fully independent — decay depends only
           on time, freq_boost depends only on reinforcements.
        2. Scores are normalized by the best node's raw score so that
           the cap at 1.0 never flattens differences between nodes.
        3. freq_boost is linear (not log) so r=3 clearly beats r=1.

        Formula:
            raw = exp(-hours / (24 * stability)) * (1 + factor * r) * conflict
            w   = raw / max(raw_scores)   ← relative normalization
        """
        cfg = self.config
        raw_scores: dict[int, float] = {}

        for vec_id, node in idx.nodes.items():
            hours_ago = (now - node.timestamp) / 3600.0

            # Decay — purely time-based, stability is a fixed config param
            decay = math.exp(
                -hours_ago / (24.0 * cfg.decay_stability_factor)
            )

            # Freq boost — linear on reinforcements (not log)
            # r=1 → 1.5x, r=3 → 2.5x, r=10 → 6.0x
            freq_boost = 1.0 + cfg.freq_boost_factor * node.reinforcements

            # Conflict penalty
            conflict_penalty = max(
                0.0, 1.0 - cfg.conflict_penalty_factor * node.conflict_score
            )

            raw_scores[vec_id] = decay * freq_boost * conflict_penalty

        if not raw_scores:
            return {}, []

        # Normalize relative to the best node in this user's index
        # This ensures the winner always gets bio_weight=1.0 and others
        # are proportional — the cap at 1.0 never flattens the signal.
        max_score = max(raw_scores.values())
        if max_score > 0:
            new_weights = {
                vid: float(s / max_score)
                for vid, s in raw_scores.items()
            }
        else:
            new_weights = {vid: 0.0 for vid in raw_scores}

        # Prune nodes below threshold AND older than 24h
        # (safety: never prune very recent nodes even if score is low)
        to_prune = [
            vid for vid, w in new_weights.items()
            if w < cfg.prune_threshold
            and (now - idx.nodes[vid].timestamp) > 24 * 3600.0
        ]

        return new_weights, to_prune

    # ── Episodic merge ────────────────────────────────────────────────

    def _merge_episodic(
        self, segment, user_id: str, idx, now: float
    ) -> int:
        """
        Detects clusters of near-duplicate episodic nodes and fuses them
        into a single semantic node (centroid).

        Episodic nodes: reinforcements <= 2 (seen only once or twice)
        Semantic threshold: cosine_sim > merge_threshold

        Returns the number of merges performed.
        """
        cfg = self.config
        merged_count = 0

        # Collect candidates — low-reinforcement nodes only
        candidates = [
            (vec_id, node)
            for vec_id, node in idx.nodes.items()
            if node.reinforcements <= 2
        ]

        if len(candidates) < 2:
            return 0

        visited = set()

        for i, (vid_a, node_a) in enumerate(candidates):
            if vid_a in visited or merged_count >= cfg.max_merges_per_cycle:
                break

            vec_a = self._get_vec(idx, node_a)
            if vec_a is None:
                continue

            cluster      = [vid_a]
            cluster_vecs = [vec_a]

            for vid_b, node_b in candidates[i + 1:]:
                if vid_b in visited:
                    continue
                vec_b = self._get_vec(idx, node_b)
                if vec_b is None:
                    continue

                sim = float(np.dot(vec_a, vec_b))
                if sim >= cfg.merge_threshold:
                    cluster.append(vid_b)
                    cluster_vecs.append(vec_b)

            if len(cluster) < 2:
                continue

            # Fuse — centroid vector, sum reinforcements
            centroid = np.mean(
                np.stack(cluster_vecs), axis=0
            ).astype(np.float32)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid /= norm

            total_reinforcements = sum(
                idx.nodes[vid].reinforcements
                for vid in cluster
                if vid in idx.nodes
            )
            latest_ts = max(
                idx.nodes[vid].timestamp
                for vid in cluster
                if vid in idx.nodes
            )

            # Pick representative meta from the most recent node
            best_vid = max(
                cluster,
                key=lambda v: idx.nodes[v].timestamp if v in idx.nodes else 0
            )
            best_meta = segment._metadata.get(user_id, {}).get(best_vid, {})

            # Delete episodic originals
            for vid in cluster:
                segment.delete(user_id, vid)
                visited.add(vid)

            # Insert semantic node
            new_meta = {
                **best_meta,
                "metadata": {
                    **(best_meta.get("metadata") or {}),
                    "timestamp":    latest_ts,
                    "reinforcements": total_reinforcements,
                    "semantic":     True,
                    "merged_from":  cluster,
                }
            }
            new_id = segment.insert(user_id, centroid, new_meta)

            # Set bio fields on the new semantic node
            new_idx = segment._indexes.get(user_id)
            if new_idx:
                new_node = new_idx.nodes.get(new_id)
                if new_node:
                    new_node.timestamp      = latest_ts
                    new_node.reinforcements = total_reinforcements
                    new_node.bio_weight     = 1.0  # semantic nodes start fresh

            merged_count += 1

        return merged_count

    def _get_vec(self, idx, node) -> np.ndarray | None:
        if node.vector is not None:
            return node.vector
        if node.quantized is not None and idx.quantizer is not None:
            v    = idx.quantizer.decode(node.quantized)
            norm = float(np.linalg.norm(v))
            return v / norm if norm > 0 else v
        return None

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"Consolidator("
            f"cycle={self._cycle_count}, "
            f"decay_stability={cfg.decay_stability_factor}, "
            f"freq_boost={cfg.freq_boost_factor}, "
            f"prune_threshold={cfg.prune_threshold}, "
            f"every={cfg.consolidate_every})"
        )