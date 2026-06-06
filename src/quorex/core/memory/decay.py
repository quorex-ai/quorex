"""
quorex.core.memory.decay
-------------------------
Temporal decay functions for memory scoring.

Transforms a cold vector database into human-like memory by
weighting old memories less than recent ones.

Score = cosine_similarity × decay(Δt)

Three strategies:
- exponential : rapid decay, good for volatile preferences / conversations
- linear      : slow decay, good for stable facts
- step        : tiered decay (short / medium / long term memory)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class DecayStrategy(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR      = "linear"
    STEP        = "step"


@dataclass
class DecayConfig:
    """
    Configuration for temporal decay.

    strategy      : which decay function to use
    half_life_h   : time (hours) after which memory weight = 0.5
                    Only used by exponential and linear strategies.
                    Default: 24h (a memory from yesterday = half weight)
    min_score     : floor — old memories never go fully to 0
                    Prevents ancient but highly similar memories from
                    being completely ignored.
    step_tiers    : for STEP strategy only.
                    List of (threshold_hours, weight) tuples, sorted asc.
                    e.g. [(1, 1.0), (24, 0.7), (168, 0.4), (720, 0.1)]
                    means: <1h=1.0, <24h=0.7, <7d=0.4, <30d=0.1, else=min
    """
    strategy    : DecayStrategy = DecayStrategy.EXPONENTIAL
    half_life_h : float         = 24.0
    min_score   : float         = 0.05
    step_tiers  : list[tuple[float, float]] | None = None

    def __post_init__(self):
        if self.strategy == DecayStrategy.STEP and self.step_tiers is None:
            self.step_tiers = [
                (1.0,   1.00),   # < 1 hour   → full weight
                (24.0,  0.75),   # < 1 day    → 75%
                (168.0, 0.50),   # < 1 week   → 50%
                (720.0, 0.20),   # < 30 days  → 20%
            ]


class TemporalDecay:
    """
    Computes temporal decay weights for stored memories.

    Usage:
        decay = TemporalDecay(DecayConfig(strategy="exponential", half_life_h=24))
        weight = decay.compute(timestamp)           # 0.0 – 1.0
        score  = decay.apply(cosine_sim, timestamp) # weighted score
    """

    def __init__(self, config: DecayConfig | None = None):
        self.config = config or DecayConfig()
        self._fn: Callable[[float], float] = self._build_fn()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, timestamp: float, now: float | None = None) -> float:
        """
        Returns a decay weight in [min_score, 1.0] for a memory
        stored at `timestamp` (unix seconds).

        timestamp : when the memory was created/last updated
        now       : reference time (defaults to current time)
        """
        now = now or time.time()
        delta_h = max(0.0, (now - timestamp) / 3600.0)
        raw = self._fn(delta_h)
        return max(self.config.min_score, min(1.0, raw))

    def apply(
        self,
        cosine_sim: float,
        timestamp: float,
        now: float | None = None,
    ) -> float:
        """
        Applies temporal decay to a cosine similarity score.

        final_score = cosine_sim × decay_weight

        cosine_sim : raw similarity score (−1 to 1, typically 0–1)
        timestamp  : unix timestamp of the stored memory
        """
        weight = self.compute(timestamp, now)
        return cosine_sim * weight

    def apply_batch(
        self,
        scores: list[float],
        timestamps: list[float],
        now: float | None = None,
    ) -> list[float]:
        """
        Applies decay to a list of (score, timestamp) pairs.
        Returns re-ranked list of weighted scores.
        """
        now = now or time.time()
        return [
            self.apply(s, t, now)
            for s, t in zip(scores, timestamps)
        ]

    # ------------------------------------------------------------------
    # Strategy builders
    # ------------------------------------------------------------------

    def _build_fn(self) -> Callable[[float], float]:
        cfg = self.config

        if cfg.strategy == DecayStrategy.EXPONENTIAL:
            # f(Δt) = 2^(−Δt / half_life)
            # At Δt=0 → 1.0
            # At Δt=half_life → 0.5
            # At Δt=∞ → 0.0 (clamped to min_score)
            lam = math.log(2) / cfg.half_life_h
            return lambda dt: math.exp(-lam * dt)

        elif cfg.strategy == DecayStrategy.LINEAR:
            # f(Δt) = max(0, 1 − Δt / (2 × half_life))
            # Reaches 0 at 2 × half_life
            cutoff = 2.0 * cfg.half_life_h
            return lambda dt: max(0.0, 1.0 - dt / cutoff)

        elif cfg.strategy == DecayStrategy.STEP:
            tiers = sorted(cfg.step_tiers, key=lambda x: x[0])

            def step_fn(dt: float) -> float:
                for threshold_h, weight in tiers:
                    if dt < threshold_h:
                        return weight
                return cfg.min_score

            return step_fn

        else:
            raise ValueError(f"Unknown decay strategy: {cfg.strategy}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def preview(self, hours: list[float] | None = None) -> dict[float, float]:
        """
        Returns a dict of {hours_ago: decay_weight} for inspection.
        Useful for debugging and tuning half_life.
        """
        hours = hours or [0, 0.5, 1, 3, 6, 12, 24, 48, 72, 168, 336, 720]
        now = time.time()
        return {
            h: round(self.compute(now - h * 3600, now), 4)
            for h in hours
        }

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"TemporalDecay("
            f"strategy={cfg.strategy}, "
            f"half_life={cfg.half_life_h}h, "
            f"min={cfg.min_score})"
        )


# ------------------------------------------------------------------
# Preset configs
# ------------------------------------------------------------------

DECAY_CONVERSATION = DecayConfig(
    strategy=DecayStrategy.EXPONENTIAL,
    half_life_h=1.0,    # Conversations decay fast — 1h half-life
    min_score=0.02,
)

DECAY_PREFERENCE = DecayConfig(
    strategy=DecayStrategy.EXPONENTIAL,
    half_life_h=168.0,  # User preferences stable for ~1 week
    min_score=0.10,
)

DECAY_FACT = DecayConfig(
    strategy=DecayStrategy.LINEAR,
    half_life_h=720.0,  # Hard facts decay over ~60 days
    min_score=0.15,
)

DECAY_HUMAN = DecayConfig(
    strategy=DecayStrategy.STEP,
    step_tiers=[
        (1.0,   1.00),  # < 1h    → full
        (24.0,  0.80),  # < 1 day → 80%
        (168.0, 0.55),  # < 1 week→ 55%
        (720.0, 0.25),  # < 30d   → 25%
    ],
    min_score=0.05,
)


if __name__ == "__main__":
    print("=== Exponential decay (half-life 24h) ===")
    d = TemporalDecay(DecayConfig(strategy=DecayStrategy.EXPONENTIAL, half_life_h=24))
    for h, w in d.preview().items():
        bar = "█" * int(w * 30)
        print(f"  {h:>6.1f}h ago → {w:.4f}  {bar}")

    print("\n=== Linear decay (half-life 24h) ===")
    d2 = TemporalDecay(DecayConfig(strategy=DecayStrategy.LINEAR, half_life_h=24))
    for h, w in d2.preview().items():
        bar = "█" * int(w * 30)
        print(f"  {h:>6.1f}h ago → {w:.4f}  {bar}")

    print("\n=== Step decay (human memory) ===")
    d3 = TemporalDecay(DECAY_HUMAN)
    for h, w in d3.preview().items():
        bar = "█" * int(w * 30)
        print(f"  {h:>6.1f}h ago → {w:.4f}  {bar}")

    print("\n=== Apply to cosine scores ===")
    import time
    now = time.time()
    memories = [
        ("2 min ago",   now - 120,      0.75),
        ("1 hour ago",  now - 3600,     0.75),
        ("1 day ago",   now - 86400,    0.75),
        ("1 week ago",  now - 604800,   0.75),
        ("30 days ago", now - 2592000,  0.75),
    ]

    d4 = TemporalDecay(DECAY_HUMAN)
    print(f"  {'Memory':<15} {'Cosine':>8} {'Decay':>8} {'Final':>8}")
    print(f"  {'-'*44}")
    for label, ts, cos in memories:
        decay = d4.compute(ts, now)
        final = d4.apply(cos, ts, now)
        print(f"  {label:<15} {cos:>8.3f} {decay:>8.3f} {final:>8.3f}")