"""
quorex.core.memory.manager
----------------------------
High-level memory interface for Quorex.

Exposes two simple methods:
    manager.remember(user_id, event)   → stores a memory
    manager.recall(user_id, query)     → retrieves ranked memories

Internally orchestrates:
    VectorDBEngine  → ANN search + WAL + snapshots
    Encoder         → text → vector
    MemoryScorer    → cosine × decay × frequency re-ranking
    Reinforcement   → increments counter when same memory is recalled
"""

from __future__ import annotations

import time
import os
from dataclasses import dataclass, field

from ..vectordb.engine import VectorDBEngine
from ..embeddings.encoder import Encoder
from .scorer import MemoryScorer, MemoryScore, ScorerConfig
from .decay import DecayConfig, DECAY_HUMAN


@dataclass
class MemoryConfig:
    """
    Full configuration for the MemoryManager.

    db_path          : directory for WAL + snapshots
    encoder_path     : directory for saved encoder state
    n_components     : SVD dimensions for embeddings
    scorer           : scorer configuration (decay + freq)
    top_k            : default number of memories to return
    threshold        : minimum final score to return a memory
    reinforce        : auto-increment reinforcement count on recall
    checkpoint_every : auto-checkpoint every N inserts
    """
    db_path          : str         = "/tmp/quorex_memory"
    encoder_path     : str         = "/tmp/quorex_encoder"
    n_components     : int         = 32
    scorer           : ScorerConfig = field(default_factory=ScorerConfig)
    top_k            : int         = 5
    threshold        : float       = 0.05
    reinforce        : bool        = True
    checkpoint_every : int         = 50


@dataclass
class Memory:
    """
    A recalled memory with full context.
    """
    vec_id          : int
    user_id         : str
    text            : str
    action          : str
    timestamp       : float
    reinforcements  : int
    final_score     : float
    cosine_sim      : float
    decay_weight    : float
    freq_weight     : float
    hours_ago       : float
    meta            : dict

    @property
    def age_str(self) -> str:
        h = self.hours_ago
        if h < 1:
            return f"{int(h * 60)}m ago"
        elif h < 24:
            return f"{h:.1f}h ago"
        elif h < 168:
            return f"{h/24:.1f}d ago"
        else:
            return f"{h/168:.1f}w ago"

    def __repr__(self) -> str:
        return (
            f"Memory("
            f"score={self.final_score:.3f}, "
            f"age={self.age_str}, "
            f"×{self.reinforcements}, "
            f"text={self.text[:40]!r})"
        )


class MemoryManager:
    """
    The core interface of Quorex.

    Transforms a cold vector database into human-like memory by combining
    semantic search with temporal decay and frequency reinforcement.

    Usage:
        manager = MemoryManager()
        manager.start(seed_events)

        manager.remember("user_123", {
            "action": "viewed_pricing",
            "metadata": {"text": "I switched to Vue.js", "plan": "pro"}
        })

        memories = manager.recall("user_123", "what framework does the user prefer?")
        for m in memories:
            print(m)

        manager.stop()
    """

    def __init__(self, config: MemoryConfig | None = None):
        self.config  = config or MemoryConfig()
        self.encoder = Encoder(n_components=self.config.n_components)
        self.engine  = VectorDBEngine(
            path=self.config.db_path,
            dim=self.config.n_components,
            checkpoint_every=self.config.checkpoint_every,
        )
        self.scorer  = MemoryScorer(self.config.scorer)
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, seed_events: list[dict] | None = None) -> None:
        """
        Starts the memory manager.
        Loads encoder from disk if available, otherwise fits on seed_events.
        Starts the vector DB engine (loads snapshot + replays WAL).
        """
        encoder_vpath = os.path.join(self.config.encoder_path, "vectorizer.json")

        if os.path.exists(encoder_vpath):
            self.encoder.load(self.config.encoder_path)
            print(f"Encoder loaded — vocab: {len(self.encoder.vectorizer.vocabulary)}")
        elif seed_events:
            self.encoder.fit(seed_events)
            self.encoder.save(self.config.encoder_path)
            print(f"Encoder fitted — vocab: {len(self.encoder.vectorizer.vocabulary)}")
        else:
            raise RuntimeError(
                "No encoder found and no seed_events provided. "
                "Pass seed_events on first start."
            )

        self.engine.start()
        self._started = True
        print(f"MemoryManager started — {self.engine.stats()['total_vectors']} memories loaded.")

    def stop(self) -> None:
        """Checkpoints and stops the engine."""
        self.engine.stop()
        self._started = False
        print("MemoryManager stopped.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def remember(
        self,
        user_id: str,
        event: dict,
        timestamp: float | None = None,
    ) -> int:
        """
        Stores a memory for a user.

        user_id   : unique user identifier
        event     : dict with 'action' + 'metadata' keys
        timestamp : unix timestamp (defaults to now)

        Returns the assigned vector id.
        """
        self._check_started()
        timestamp = timestamp or time.time()

        # Inject timestamp + reinforcement counter into metadata
        meta = dict(event)
        if "metadata" not in meta:
            meta["metadata"] = {}
        meta["metadata"]["timestamp"]      = timestamp
        meta["metadata"]["reinforcements"] = 1

        # Encode
        vector = self.encoder.encode(event)

        # Store
        vec_id = self.engine.insert(user_id, vector, meta)
        return vec_id

    def recall(
        self,
        user_id: str,
        query: str | dict,
        top_k: int | None = None,
        threshold: float | None = None,
        now: float | None = None,
    ) -> list[Memory]:
        """
        Retrieves the most relevant memories for a user.

        query     : raw text string or event dict
        top_k     : max memories to return (default: config.top_k)
        threshold : min final score (default: config.threshold)
        now       : reference time for decay (default: current time)

        Returns list of Memory objects, ranked by final score.
        """
        self._check_started()
        top_k     = top_k     or self.config.top_k
        threshold = threshold or self.config.threshold
        now       = now       or time.time()

        # Encode query
        if isinstance(query, str):
            query_vec = self.encoder.encode_text(query)
        else:
            query_vec = self.encoder.encode(query)

        # ANN search — fetch more than top_k for re-ranking
        raw_results = self.engine.search(
            user_id,
            query_vec,
            top_k=top_k * 3,
        )

        if not raw_results:
            return []

        # Normalize result format for scorer
        scorer_input = []
        for r in raw_results:
            meta = r.get("meta", {})
            inner_meta = meta.get("metadata", {})
            scorer_input.append({
                "id":    r["id"],
                "score": r["score"],
                "meta":  {
                    **meta,
                    "timestamp":     inner_meta.get("timestamp", now),
                    "reinforcements": inner_meta.get("reinforcements", 1),
                }
            })

        # Re-rank with temporal decay + frequency
        ranked: list[MemoryScore] = self.scorer.rank(
            scorer_input,
            now=now,
            top_k=top_k,
            threshold=threshold,
        )

        # Reinforce recalled memories
        if self.config.reinforce:
            for ms in ranked:
                self._reinforce(user_id, ms.vec_id)

        # Convert to Memory objects
        return [self._to_memory(user_id, ms) for ms in ranked]

    def forget(self, user_id: str, vec_id: int) -> bool:
        """
        Soft-deletes a memory (GDPR right to erasure).
        Returns True if deleted, False if not found.
        """
        self._check_started()
        return self.engine.delete(user_id, vec_id)

    def purge(self, user_id: str) -> int:
        """
        Deletes ALL memories for a user.
        Returns number of memories deleted.
        """
        self._check_started()
        seg = self.engine.segment
        if user_id not in seg._metadata:
            return 0

        vec_ids = list(seg._metadata[user_id].keys())
        for vid in vec_ids:
            self.engine.delete(user_id, vid)

        return len(vec_ids)

    def stats(self, user_id: str | None = None) -> dict:
        """
        Returns memory statistics.
        If user_id provided, returns per-user stats.
        """
        self._check_started()
        engine_stats = self.engine.stats()

        if user_id:
            seg = self.engine.segment
            user_count = seg.vector_count(user_id) if seg.has_user(user_id) else 0
            return {
                "user_id":        user_id,
                "memories":       user_count,
                "encoder_vocab":  len(self.encoder.vectorizer.vocabulary),
                "encoder_dims":   self.encoder.reducer.n_components,
            }

        return {
            **engine_stats,
            "encoder_vocab": len(self.encoder.vectorizer.vocabulary),
            "encoder_dims":  self.encoder.reducer.n_components,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reinforce(self, user_id: str, vec_id: int) -> None:
        """Increments the reinforcement counter for a recalled memory."""
        seg = self.engine.segment
        if user_id not in seg._metadata:
            return
        if vec_id not in seg._metadata[user_id]:
            return

        meta = seg._metadata[user_id][vec_id]
        inner = meta.get("metadata", {})
        inner["reinforcements"] = inner.get("reinforcements", 1) + 1
        meta["metadata"] = inner

    def _to_memory(self, user_id: str, ms: MemoryScore) -> Memory:
        """Converts a MemoryScore into a Memory object."""
        meta      = ms.meta
        inner     = meta.get("metadata", {})
        text      = inner.get("text", meta.get("action", ""))
        action    = meta.get("action", "")
        timestamp = inner.get("timestamp", time.time())

        return Memory(
            vec_id=ms.vec_id,
            user_id=user_id,
            text=text,
            action=action,
            timestamp=timestamp,
            reinforcements=ms.reinforcements,
            final_score=ms.final_score,
            cosine_sim=ms.cosine_sim,
            decay_weight=ms.decay_weight,
            freq_weight=ms.freq_weight,
            hours_ago=ms.hours_ago,
            meta=meta,
        )

    def _check_started(self) -> None:
        if not self._started:
            raise RuntimeError("MemoryManager not started. Call start() first.")

    def __repr__(self) -> str:
        status = "running" if self._started else "stopped"
        return (
            f"MemoryManager("
            f"status={status}, "
            f"vocab={len(self.encoder.vectorizer.vocabulary)}, "
            f"dims={self.encoder.reducer.n_components})"
        )


if __name__ == "__main__":
    import shutil

    DB_PATH      = "/tmp/quorex_manager_test"
    ENCODER_PATH = "/tmp/quorex_encoder_test"

    shutil.rmtree(DB_PATH, ignore_errors=True)
    shutil.rmtree(ENCODER_PATH, ignore_errors=True)

    SEED = [
        {"action": "viewed pricing page", "metadata": {"plan": "pro", "text": "viewed pricing"}},
        {"action": "searched pricing", "metadata": {"text": "searched pricing"}},
        {"action": "upgraded plan pro", "metadata": {"text": "upgraded to pro"}},
        {"action": "visited homepage", "metadata": {"text": "visited homepage"}},
        {"action": "searched docs api", "metadata": {"text": "searched docs"}},
        {"action": "clicked cta dashboard", "metadata": {"text": "clicked cta"}},
        {"action": "write code python", "metadata": {"text": "coding python"}},
        {"action": "read book novel story", "metadata": {"text": "reading"}},
        {"action": "play guitar music", "metadata": {"text": "playing music"}},
        {"action": "drive car road", "metadata": {"text": "driving"}},
    ]

    config = MemoryConfig(
        db_path=DB_PATH,
        encoder_path=ENCODER_PATH,
        n_components=8,
        top_k=3,
        threshold=0.01,
    )

    now = time.time()

    with MemoryManager(config) as manager:
        manager.start(SEED)

        print("\n=== Storing memories ===")
        manager.remember("user_123", {
            "action": "viewed pricing page",
            "metadata": {"text": "I checked the pricing page", "plan": "pro"}
        }, timestamp=now - 7200)       # 2 hours ago

        manager.remember("user_123", {
            "action": "searched pricing",
            "metadata": {"text": "Searching for pricing info"}
        }, timestamp=now - 3600)       # 1 hour ago

        manager.remember("user_123", {
            "action": "upgraded plan pro",
            "metadata": {"text": "Just upgraded to Pro plan!"}
        }, timestamp=now - 300)        # 5 minutes ago

        manager.remember("user_123", {
            "action": "write code python",
            "metadata": {"text": "Working on a Python script"}
        }, timestamp=now - 86400)      # 1 day ago — different topic

        print("\n=== Recalling memories ===")
        memories = manager.recall("user_123", "what did the user do with pricing?")

        print(f"\n{'#':<4} {'Text':<35} {'Score':>7} {'Cos':>6} {'Decay':>7} {'Freq':>6} {'Age':>10}")
        print("─" * 80)
        for i, m in enumerate(memories):
            print(
                f"#{i+1:<3} {m.text[:33]:<35} {m.final_score:>7.3f} "
                f"{m.cosine_sim:>6.3f} {m.decay_weight:>7.3f} "
                f"{m.freq_weight:>6.3f} {m.age_str:>10}"
            )

        print(f"\n=== Stats ===")
        print(manager.stats("user_123"))
        print(manager)