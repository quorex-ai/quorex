"""
quorex.core.memory.scorer
--------------------------
Combines multiple signals into a single memory relevance score.

Final score = cosine_sim × decay(Δt) × frequency_boost(n)

Three signals:
1. Semantic similarity  : how relevant is the memory to the query
2. Temporal decay       : how recent is the memory
3. Frequency boost      : how often was this memory reinforced

This transforms raw vector search into human-like memory recall —
recent, frequently-reinforced, relevant memories surface first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .decay import DecayConfig, DecayStrategy, TemporalDecay, DECAY_HUMAN


@dataclass
class ScorerConfig:
    """
    Configuration for the memory scorer.

    decay           : temporal decay configuration
    freq_boost      : whether to apply frequency boosting
    freq_max_boost  : maximum multiplier from frequency (e.g. 1.3 = +30%)
    freq_saturation : number of reinforcements at which boost maxes out
    weights         : (semantic, temporal, frequency) — must sum to 1.0
                      Used for weighted combination mode.
    mode            : "multiplicative" (default) or "weighted"
                      multiplicative: score = cos × decay × freq_boost
                      weighted: score = w1×cos + w2×decay + w3×freq_norm
    """
    decay           : DecayConfig = field(default_factory=lambda: DECAY_HUMAN)
    freq_boost      : bool        = True
    freq_max_boost  : float       = 1.30
    freq_saturation : int         = 10
    weights         : tuple[float, float, float] = (0.6, 0.3, 0.1)
    mode            : str         = "multiplicative"


@dataclass
class MemoryScore:
    """
    Detailed breakdown of a memory's relevance score.
    """
    vec_id          : int
    final_score     : float
    cosine_sim      : float
    decay_weight    : float
    freq_weight     : float
    reinforcements  : int
    hours_ago       : float
    meta            : dict

    def __repr__(self) -> str:
        return (
            f"MemoryScore("
            f"id={self.vec_id}, "
            f"final={self.final_score:.4f}, "
            f"cos={self.cosine_sim:.4f}, "
            f"decay={self.decay_weight:.4f}, "
            f"freq={self.freq_weight:.4f}, "
            f"age={self.hours_ago:.1f}h)"
        )


class MemoryScorer:
    """
    Scores and re-ranks memories using semantic + temporal + frequency signals.

    Usage:
        scorer = MemoryScorer()
        ranked = scorer.rank(raw_results, now=time.time())
    """

    def __init__(self, config: ScorerConfig | None = None):
        self.config = config or ScorerConfig()
        self.decay = TemporalDecay(self.config.decay)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        cosine_sim: float,
        timestamp: float,
        reinforcements: int = 1,
        now: float | None = None,
    ) -> tuple[float, float, float]:
        """
        Computes final score for a single memory.

        Returns (final_score, decay_weight, freq_weight).
        """
        now = now or time.time()
        cfg = self.config

        # 1 — Temporal decay
        decay_w = self.decay.compute(timestamp, now)

        # 2 — Frequency boost
        freq_w = self._freq_boost(reinforcements)

        # 3 — Combine
        if cfg.mode == "multiplicative":
            final = cosine_sim * decay_w * freq_w
        else:
            w1, w2, w3 = cfg.weights
            freq_norm = (freq_w - 1.0) / (cfg.freq_max_boost - 1.0 + 1e-9)
            final = w1 * cosine_sim + w2 * decay_w + w3 * freq_norm

        return final, decay_w, freq_w

    def rank(
        self,
        results: list[dict],
        now: float | None = None,
        top_k: int | None = None,
        threshold: float = 0.0,
    ) -> list[MemoryScore]:
        """
        Re-ranks a list of raw search results using temporal + frequency signals.

        Each result dict must contain:
            id       : int   — vector id
            score    : float — cosine similarity
            meta     : dict  — must contain 'timestamp' (unix seconds)
                               optionally 'reinforcements' (int, default 1)

        Returns sorted list of MemoryScore objects (highest first).
        """
        now = now or time.time()
        scored = []

        for r in results:
            meta = r.get("meta", {})
            vec_id = r.get("id", -1)
            cosine_sim = r.get("score", 0.0)
            timestamp = meta.get("timestamp", now)
            reinforcements = meta.get("reinforcements", 1)

            final, decay_w, freq_w = self.score(
                cosine_sim, timestamp, reinforcements, now
            )

            if final < threshold:
                continue

            hours_ago = (now - timestamp) / 3600.0

            scored.append(MemoryScore(
                vec_id=vec_id,
                final_score=round(final, 6),
                cosine_sim=round(cosine_sim, 6),
                decay_weight=round(decay_w, 6),
                freq_weight=round(freq_w, 6),
                reinforcements=reinforcements,
                hours_ago=round(hours_ago, 2),
                meta=meta,
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored

    def explain(
        self,
        cosine_sim: float,
        timestamp: float,
        reinforcements: int = 1,
        now: float | None = None,
    ) -> str:
        """
        Returns a human-readable explanation of how a score was computed.
        Useful for debugging and transparency.
        """
        now = now or time.time()
        final, decay_w, freq_w = self.score(
            cosine_sim, timestamp, reinforcements, now
        )
        hours_ago = (now - timestamp) / 3600.0

        lines = [
            f"Score breakdown:",
            f"  Cosine similarity : {cosine_sim:.4f}",
            f"  Temporal decay    : {decay_w:.4f}  ({hours_ago:.1f}h ago)",
            f"  Frequency boost   : {freq_w:.4f}  ({reinforcements} reinforcements)",
            f"  ─────────────────────────────",
            f"  Final score       : {final:.4f}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _freq_boost(self, n: int) -> float:
        """
        Frequency boost: log-scaled, saturates at freq_saturation.

        f(n) = 1 + (max_boost - 1) × log(1 + n) / log(1 + saturation)

        f(1)           = 1.0    (baseline, no boost)
        f(saturation)  ≈ max_boost
        f(∞)           → max_boost (capped)
        """
        if not self.config.freq_boost:
            return 1.0

        import math
        cfg = self.config
        n = max(1, n)
        boost = 1.0 + (cfg.freq_max_boost - 1.0) * (
            math.log(1 + n) / math.log(1 + cfg.freq_saturation)
        )
        return min(boost, cfg.freq_max_boost)

    def __repr__(self) -> str:
        return (
            f"MemoryScorer("
            f"mode={self.config.mode}, "
            f"decay={self.config.decay.strategy}, "
            f"freq_boost={self.config.freq_boost})"
        )


if __name__ == "__main__":
    import time

    now = time.time()
    scorer = MemoryScorer()

    print("=== Memory Scorer Demo ===\n")
    print(f"{scorer}\n")

    # Simulate 5 memories with same cosine similarity but different age + frequency
    memories = [
        {
            "id": 0,
            "score": 0.82,
            "meta": {
                "timestamp": now - 60,          # 1 min ago
                "reinforcements": 1,
                "text": "I switched to Vue.js"
            }
        },
        {
            "id": 1,
            "score": 0.80,
            "meta": {
                "timestamp": now - 3600,         # 1 hour ago
                "reinforcements": 5,             # mentioned 5 times
                "text": "I prefer dark mode"
            }
        },
        {
            "id": 2,
            "score": 0.85,
            "meta": {
                "timestamp": now - 86400,        # 1 day ago
                "reinforcements": 1,
                "text": "I code in Python"
            }
        },
        {
            "id": 3,
            "score": 0.78,
            "meta": {
                "timestamp": now - 604800,       # 1 week ago
                "reinforcements": 8,             # mentioned 8 times
                "text": "I use React for frontend"
            }
        },
        {
            "id": 4,
            "score": 0.91,
            "meta": {
                "timestamp": now - 2592000,      # 30 days ago
                "reinforcements": 2,
                "text": "I work at a startup"
            }
        },
    ]

    print(f"{'#':<4} {'Text':<30} {'Cosine':>7} {'Decay':>7} {'Freq':>7} {'Final':>7} {'Age':>10}")
    print(f"{'─' * 78}")

    ranked = scorer.rank(memories, now=now)
    for i, ms in enumerate(ranked):
        text = ms.meta.get("text", "")[:28]
        age = f"{ms.hours_ago:.1f}h"
        print(
            f"#{i+1:<3} {text:<30} {ms.cosine_sim:>7.3f} "
            f"{ms.decay_weight:>7.3f} {ms.freq_weight:>7.3f} "
            f"{ms.final_score:>7.3f} {age:>10}"
        )

    print(f"\n=== Score explanation for memory #3 (week-old, freq=8) ===")
    m = memories[3]
    print(scorer.explain(
        m["score"],
        m["meta"]["timestamp"],
        m["meta"]["reinforcements"],
        now
    ))

    print(f"\n=== Frequency boost curve ===")
    print(f"  {'Reinforcements':<20} {'Boost':>8}")
    print(f"  {'─' * 30}")
    for n in [1, 2, 3, 5, 8, 10, 15, 20]:
        boost = scorer._freq_boost(n)
        bar = "█" * int((boost - 1.0) * 30 / 0.3)
        print(f"  {n:<20} {boost:>8.4f}  {bar}")