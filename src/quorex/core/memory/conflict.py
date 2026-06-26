"""
quorex.core.memory.conflict
-----------------------------
Conflict detection and resolution for Quorex memory.

When a user says "I switched from React to Vue.js", Quorex must:
1. Detect that the new memory conflicts with an existing one
2. Archive the old memory (keep it for history but suppress it)
3. Activate the new memory with full weight

Two types of memory updates:
- REINFORCEMENT : same info repeated → increment reinforcement counter
- CONFLICT      : contradictory info → archive old, activate new

Conflict detection uses cosine similarity threshold on the same
semantic category. High similarity + different content = conflict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..vectordb.engine import VectorDBEngine
    from ..embeddings.encoder import Encoder


class ConflictType(str, Enum):
    NONE          = "none"           # No conflict — new memory
    REINFORCEMENT = "reinforcement"  # Same info repeated
    CONFLICT      = "conflict"       # Contradictory info detected


@dataclass
class ConflictResult:
    """
    Result of conflict detection for a new memory.
    """
    conflict_type    : ConflictType
    new_vec_id       : int | None = None
    archived_vec_ids : list[int]  = field(default_factory=list)
    reinforced_vec_id: int | None = None
    similarity       : float      = 0.0
    message          : str        = ""

    @property
    def is_conflict(self) -> bool:
        return self.conflict_type == ConflictType.CONFLICT

    @property
    def is_reinforcement(self) -> bool:
        return self.conflict_type == ConflictType.REINFORCEMENT

    def __repr__(self) -> str:
        return (
            f"ConflictResult("
            f"type={self.conflict_type}, "
            f"sim={self.similarity:.3f}, "
            f"archived={self.archived_vec_ids}, "
            f"msg={self.message!r})"
        )


@dataclass
class ConflictConfig:
    """
    Configuration for conflict detection.

    reinforcement_threshold : cosine similarity above which
                              a new memory is considered a reinforcement
                              of an existing one (same info, repeated).
                              Default: 0.92

    conflict_threshold      : cosine similarity above which
                              a new memory is considered to conflict
                              with an existing one (similar topic,
                              different claim).
                              Default: 0.65

    max_archived_per_slot   : max number of archived memories to keep
                              per semantic slot (old memories pruned
                              beyond this limit).
                              Default: 3

    category_field          : metadata field used to group memories
                              into semantic slots for conflict detection.
                              Default: "category"
    """
    reinforcement_threshold : float = 0.95
    conflict_threshold      : float = 0.45
    max_archived_per_slot   : int   = 3
    category_field          : str   = "category"


class ConflictResolver:
    """
    Detects and resolves memory conflicts for a given user.

    Usage:
        resolver = ConflictResolver(config)

        result = resolver.check_and_resolve(
            engine     = engine,
            encoder    = encoder,
            user_id    = "user_123",
            new_vector = vec,
            new_meta   = meta,
        )

        if result.is_conflict:
            print(f"Conflict resolved: {result.message}")
        elif result.is_reinforcement:
            print(f"Memory reinforced: {result.message}")
    """

    def __init__(self, config: ConflictConfig | None = None):
        self.config = config or ConflictConfig()
        self._conflict_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_resolve(
        self,
        engine    ,
        encoder   ,
        user_id   : str,
        new_vector: np.ndarray,
        new_meta  : dict,
    ) -> ConflictResult:
        """
        Checks if a new memory conflicts with existing ones.
        Resolves conflicts by archiving old memories.

        Returns a ConflictResult describing what happened.
        """
        seg = engine.segment
        if not seg.has_user(user_id):
            return ConflictResult(conflict_type=ConflictType.NONE)

        # Get all active (non-archived) memories for the user
        active_memories = self._get_active_memories(seg, user_id)
        if not active_memories:
            return ConflictResult(conflict_type=ConflictType.NONE)

        # Compute cosine similarities against all active memories
        similarities = self._compute_similarities(
            new_vector, active_memories, seg, user_id
        )

        if not similarities:
            return ConflictResult(conflict_type=ConflictType.NONE)

        best_vec_id, best_sim = max(similarities.items(), key=lambda x: x[1])

        # --- REINFORCEMENT ---
        if best_sim >= self.config.reinforcement_threshold:
            return self._reinforce(seg, user_id, best_vec_id, best_sim)

        # --- CONFLICT ---
        if best_sim >= self.config.conflict_threshold:
            return self._resolve_conflict(
                engine, seg, user_id,
                new_vector, new_meta,
                best_vec_id, best_sim,
                similarities,
            )

        # --- NO CONFLICT ---
        return ConflictResult(conflict_type=ConflictType.NONE)

    def get_conflict_log(self, user_id: str | None = None) -> list[dict]:
        """Returns the conflict history, optionally filtered by user."""
        if user_id:
            return [e for e in self._conflict_log if e.get("user_id") == user_id]
        return list(self._conflict_log)

    def get_archived_memories(self, seg, user_id: str) -> list[dict]:
        """Returns all archived memories for a user."""
        if user_id not in seg._metadata:
            return []
        return [
            {"vec_id": vid, **meta}
            for vid, meta in seg._metadata[user_id].items()
            if meta.get("metadata", {}).get("__status__") == "archived"
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_active_memories(self, seg, user_id: str) -> list[int]:
        """Returns vec_ids of all non-archived, non-deleted memories."""
        if user_id not in seg._metadata:
            return []
        return [
            vid for vid, meta in seg._metadata[user_id].items()
            if meta.get("metadata", {}).get("__status__", "active") == "active"
            and not meta.get("__deleted__", False)
        ]

    def _compute_similarities(
        self,
        new_vector     : np.ndarray,
        active_vec_ids : list[int],
        seg,
        user_id        : str,
    ) -> dict[int, float]:
        """
        Computes cosine similarity between new_vector and all active memories.
        Returns {vec_id: similarity}.
        """
        idx = seg._indexes.get(user_id)
        if idx is None:
            return {}

        similarities = {}
        new_norm = np.linalg.norm(new_vector)
        if new_norm == 0:
            return {}
        new_unit = new_vector / new_norm

        for vid in active_vec_ids:
            if vid not in idx.nodes:
                continue
            stored_vec = idx.nodes[vid].vector  # already normalized in HNSW
            sim = float(np.dot(new_unit, stored_vec))
            similarities[vid] = max(0.0, sim)

        return similarities

    def _reinforce(
        self,
        seg      ,
        user_id  : str,
        vec_id   : int,
        sim      : float,
    ) -> ConflictResult:
        """Increments reinforcement counter for an existing memory."""
        meta  = seg._metadata[user_id][vec_id]
        inner = meta.get("metadata", {})
        count = inner.get("reinforcements", 1) + 1
        inner["reinforcements"] = count
        inner["last_reinforced"] = time.time()
        meta["metadata"] = inner

        msg = f"Reinforced vec_id={vec_id} (×{count}, sim={sim:.3f})"
        self._log("reinforcement", user_id, vec_id, None, sim, msg)

        return ConflictResult(
            conflict_type     = ConflictType.REINFORCEMENT,
            reinforced_vec_id = vec_id,
            similarity        = sim,
            message           = msg,
        )

    def _resolve_conflict(
        self,
        engine        ,
        seg           ,
        user_id       : str,
        new_vector    : np.ndarray,
        new_meta      : dict,
        best_vec_id   : int,
        best_sim      : float,
        all_sims      : dict[int, float],
    ) -> ConflictResult:
        """
        Archives conflicting memories and inserts the new one as active.
        """
        cfg = self.config

        # Find all memories above conflict threshold
        conflicting = [
            vid for vid, sim in all_sims.items()
            if sim >= cfg.conflict_threshold
        ]

        # Archive conflicting memories
        archived = []
        for vid in conflicting:
            meta  = seg._metadata[user_id][vid]
            inner = meta.get("metadata", {})
            old_text = inner.get("text", meta.get("action", "unknown"))

            inner["__status__"]    = "archived"
            inner["__archived_at__"] = time.time()
            inner["__archived_reason__"] = "conflict_resolved"
            meta["metadata"] = inner
            archived.append(vid)

        # Extract old text for log message
        old_meta  = seg._metadata[user_id][best_vec_id]
        old_inner = old_meta.get("metadata", {})
        old_text  = old_inner.get("text", old_meta.get("action", "?"))
        new_inner = new_meta.get("metadata", {})
        new_text  = new_inner.get("text", new_meta.get("action", "?"))

        # Mark new memory as active
        new_inner["__status__"] = "active"
        new_meta["metadata"]    = new_inner

        msg = (
            f"Conflict resolved: archived {len(archived)} memory(ies) "
            f"(sim={best_sim:.3f}) — "
            f"'{old_text[:40]}' → '{new_text[:40]}'"
        )

        self._log("conflict", user_id, best_vec_id, None, best_sim, msg)

        # Prune excess archived memories if needed
        self._prune_archived(seg, user_id)

        return ConflictResult(
            conflict_type    = ConflictType.CONFLICT,
            archived_vec_ids = archived,
            similarity       = best_sim,
            message          = msg,
        )

    def _prune_archived(self, seg, user_id: str) -> None:
        """Keeps only the N most recent archived memories per user."""
        archived = self.get_archived_memories(seg, user_id)
        if len(archived) <= self.config.max_archived_per_slot:
            return

        # Sort by archived_at ascending (oldest first)
        archived.sort(
            key=lambda x: x.get("metadata", {}).get("__archived_at__", 0)
        )

        to_prune = archived[:-self.config.max_archived_per_slot]
        for entry in to_prune:
            vid = entry["vec_id"]
            if vid in seg._metadata.get(user_id, {}):
                seg._metadata[user_id][vid]["__deleted__"] = True

    def _log(
        self,
        event_type : str,
        user_id    : str,
        old_vec_id : int | None,
        new_vec_id : int | None,
        similarity : float,
        message    : str,
    ) -> None:
        self._conflict_log.append({
            "event_type" : event_type,
            "user_id"    : user_id,
            "old_vec_id" : old_vec_id,
            "new_vec_id" : new_vec_id,
            "similarity" : round(similarity, 4),
            "message"    : message,
            "timestamp"  : time.time(),
        })

    def __repr__(self) -> str:
        return (
            f"ConflictResolver("
            f"reinforce_threshold={self.config.reinforcement_threshold}, "
            f"conflict_threshold={self.config.conflict_threshold})"
        )


if __name__ == "__main__":
    import sys, shutil
    sys.path.insert(0, ".")

    from core.embeddings.encoder import Encoder
    from core.vectordb.engine import VectorDBEngine

    shutil.rmtree("/tmp/quorex_conflict_test", ignore_errors=True)

    SEED = [
        {"action": "write code react", "metadata": {"text": "coding react"}},
        {"action": "write code vue", "metadata": {"text": "coding vue"}},
        {"action": "write code python", "metadata": {"text": "coding python"}},
        {"action": "use dark mode", "metadata": {"text": "dark mode"}},
        {"action": "use light mode", "metadata": {"text": "light mode"}},
        {"action": "play guitar music", "metadata": {"text": "music"}},
        {"action": "buy groceries market", "metadata": {"text": "shopping"}},
    ]

    encoder = Encoder(n_components=8)
    encoder.fit(SEED)

    engine = VectorDBEngine(path="/tmp/quorex_conflict_test", dim=8)
    engine.start()

    resolver = ConflictResolver()
    print(resolver)
    print()

    def store(uid, event, text, ts=None):
        ts = ts or time.time()
        meta = dict(event)
        meta.setdefault("metadata", {})
        meta["metadata"]["text"] = text
        meta["metadata"]["timestamp"] = ts
        meta["metadata"]["reinforcements"] = 1
        meta["metadata"]["__status__"] = "active"
        vec = encoder.encode(event)
        result = resolver.check_and_resolve(engine, encoder, uid, vec, meta)
        if result.conflict_type == ConflictType.NONE:
            vid = engine.insert(uid, vec, meta)
            print(f"  [NEW]         vec_id={vid} — {text}")
        elif result.is_reinforcement:
            print(f"  [REINFORCE]   {result.message}")
        elif result.is_conflict:
            vid = engine.insert(uid, vec, meta)
            print(f"  [CONFLICT]    {result.message}")
        return result

    print("=== Storing memories ===")
    store("user_123", {"action": "write code react"}, "I code in React for frontend")
    store("user_123", {"action": "use dark mode"}, "I always use dark mode")
    store("user_123", {"action": "play guitar music"}, "I play guitar in my spare time")

    print("\n=== Reinforcement (same info) ===")
    store("user_123", {"action": "use dark mode"}, "Dark mode is the only way to work")

    print("\n=== Conflict (contradictory info) ===")
    store("user_123", {"action": "write code vue"}, "I switched to Vue.js, React was too verbose")

    print("\n=== Conflict log ===")
    for entry in resolver.get_conflict_log("user_123"):
        print(f"  [{entry['event_type'].upper()}] {entry['message']}")

    print("\n=== Archived memories ===")
    for m in resolver.get_archived_memories(engine.segment, "user_123"):
        print(f"  archived: {m.get('metadata', {}).get('text', '?')}")

    engine.stop()