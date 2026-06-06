"""
quorex.core.memory.recall
--------------------------
Advanced recall system for Quorex.

Goes beyond basic ANN search with:
1. Contextual recall   — query-aware, not just similarity-based
2. Multi-signal recall — cosine × decay × frequency × recency_of_recall
3. Recall chains       — surface associated memories (graph traversal)
4. Selective recall    — filter by category, status, time window
5. Summarized recall   — condensed context ready to inject into a LLM
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class RecallMode(str, Enum):
    SEMANTIC   = "semantic"    # Pure cosine similarity (baseline)
    TEMPORAL   = "temporal"    # Decay-weighted recall
    CONTEXTUAL = "contextual"  # Full multi-signal (default)
    CHAIN      = "chain"       # Graph traversal — recall associated memories
    SUMMARY    = "summary"     # Returns condensed context string for LLM


@dataclass
class RecallFilter:
    """
    Filters to apply before scoring recalled memories.

    category       : only return memories with this category tag
    status         : "active" (default) | "archived" | "all"
    max_age_hours  : ignore memories older than N hours
    min_score      : minimum final score threshold
    user_ids       : if set, only recall from these user IDs (cross-user)
    """
    category      : str | None  = None
    status        : str         = "active"
    max_age_hours : float | None = None
    min_score     : float       = 0.0
    user_ids      : list[str] | None = None


@dataclass
class RecallConfig:
    """
    Configuration for the recall system.

    mode                  : recall strategy (see RecallMode)
    top_k                 : number of memories to return
    chain_depth           : for CHAIN mode — how many hops to traverse
    chain_sim_threshold   : min similarity to follow a chain link
    recency_boost_factor  : boost for recently recalled memories
                            (memories recalled often get a boost)
    summary_max_tokens    : approximate max chars for SUMMARY mode output
    """
    mode                 : RecallMode  = RecallMode.CONTEXTUAL
    top_k                : int         = 5
    chain_depth          : int         = 2
    chain_sim_threshold  : float       = 0.55
    recency_boost_factor : float       = 1.15
    summary_max_tokens   : int         = 800


@dataclass
class RecalledMemory:
    """A single recalled memory with full signal breakdown."""
    vec_id          : int
    user_id         : str
    text            : str
    action          : str
    category        : str
    status          : str
    timestamp       : float
    last_recalled   : float | None
    reinforcements  : int
    recall_count    : int
    final_score     : float
    cosine_sim      : float
    decay_weight    : float
    freq_weight     : float
    recency_weight  : float
    chain_depth     : int       = 0
    meta            : dict      = field(default_factory=dict)

    @property
    def hours_ago(self) -> float:
        return (time.time() - self.timestamp) / 3600.0

    @property
    def age_str(self) -> str:
        h = self.hours_ago
        if h < 1:   return f"{int(h * 60)}m ago"
        if h < 24:  return f"{h:.1f}h ago"
        if h < 168: return f"{h / 24:.1f}d ago"
        return f"{h / 168:.1f}w ago"

    def to_context_str(self) -> str:
        """Formats memory as a context string for LLM injection."""
        return f"[{self.age_str}] {self.text}"

    def __repr__(self) -> str:
        return (
            f"RecalledMemory("
            f"score={self.final_score:.3f}, "
            f"age={self.age_str}, "
            f"×{self.reinforcements}, "
            f"text={self.text[:40]!r})"
        )


class RecallEngine:
    """
    Advanced recall system combining multiple relevance signals.

    Usage:
        engine = RecallEngine(config)

        results = engine.recall(
            seg        = segment,
            encoder    = encoder,
            user_id    = "user_123",
            query      = "what framework does this user prefer?",
            filters    = RecallFilter(category="tech_stack"),
        )

        context = engine.summarize(results)
        # → inject context into LLM system prompt
    """

    def __init__(self, config: RecallConfig | None = None):
        self.config = config or RecallConfig()
        self._recall_log: dict[str, dict[int, dict]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recall(
        self,
        seg       ,
        encoder   ,
        user_id   : str,
        query     : str | np.ndarray,
        filters   : RecallFilter | None = None,
        now       : float | None = None,
    ) -> list[RecalledMemory]:
        """
        Main recall method. Combines all signals based on config.mode.
        """
        now     = now or time.time()
        filters = filters or RecallFilter()
        mode    = self.config.mode

        # Encode query
        if isinstance(query, str):
            query_vec = encoder.encode_text(query)
        else:
            query_vec = query

        # Get candidate memories
        candidates = self._get_candidates(seg, user_id, filters, now)
        if not candidates:
            return []

        # Score based on mode
        if mode == RecallMode.SEMANTIC:
            scored = self._score_semantic(candidates, query_vec, seg, user_id, now)
        elif mode == RecallMode.TEMPORAL:
            scored = self._score_temporal(candidates, query_vec, seg, user_id, now)
        elif mode == RecallMode.CHAIN:
            scored = self._score_chain(candidates, query_vec, seg, user_id, now)
        else:  # CONTEXTUAL (default)
            scored = self._score_contextual(candidates, query_vec, seg, user_id, now)

        # Apply min_score filter
        scored = [m for m in scored if m.final_score >= filters.min_score]

        # Sort and limit
        scored.sort(key=lambda x: x.final_score, reverse=True)
        top = scored[:self.config.top_k]

        # Update recall metadata
        for m in top:
            self._update_recall_log(user_id, m.vec_id, seg, now)

        return top

    def summarize(
        self,
        memories  : list[RecalledMemory],
        user_id   : str | None = None,
        max_chars : int | None = None,
    ) -> str:
        """
        Converts recalled memories into a condensed context string
        ready to inject into a LLM system prompt.

        Format:
            ## User Memory Context
            [2h ago] I prefer Vue.js over React (score: 0.82)
            [5m ago] Just upgraded to Pro plan (score: 0.91)
            ...
        """
        max_chars = max_chars or self.config.summary_max_tokens

        if not memories:
            return "## User Memory Context\n(no relevant memories found)"

        lines = ["## User Memory Context"]
        if user_id:
            lines[0] += f" — {user_id}"

        total_chars = len(lines[0])
        for m in memories:
            line = f"- [{m.age_str}] {m.text}"
            if len(line) + total_chars > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        lines.append(
            f"\n(based on {len(memories)} memories, "
            f"ranked by relevance × recency × frequency)"
        )
        return "\n".join(lines)

    def summarize_session(
        self,
        messages       : list[dict],
        encoder        ,
        min_length     : int   = 4,
        noise_threshold: float = 0.15,
        max_facts      : int   = 10,
    ) -> list[dict]:
        """
        Summarizes a raw conversation session into condensed memory facts.

        Filters noise (short/generic messages), clusters semantically
        similar messages, and returns a list of condensed memory dicts
        ready to store via remember().

        messages : list of dicts with at least a "text" key.
                   e.g. [{"text": "I code in Python"}, {"text": "ok cool"}]

        min_length      : minimum word count to consider a message non-noise
        noise_threshold : cosine similarity threshold for clustering messages
        max_facts       : maximum number of condensed facts to return

        Returns list of dicts:
        [
            {
                "action"  : "summarized_fact",
                "metadata": {
                    "text"           : "Uses Python and React as main stack",
                    "source_count"   : 3,
                    "category"       : "tech_stack",
                    "reinforcements" : 3,
                }
            },
            ...
        ]
        """
        if not messages:
            return []

        # Step 1 — Filter noise
        clean = self._filter_noise(messages, min_length)
        if not clean:
            return []

        # Step 2 — Encode all messages
        vecs = []
        for msg in clean:
            text = msg.get("text", "")
            try:
                vec = encoder.encode_text(text)
                vecs.append((text, vec, msg))
            except Exception:
                continue

        if not vecs:
            return []

        # Step 3 — Cluster by semantic similarity
        clusters = self._cluster_messages(vecs, noise_threshold)

        # Step 4 — Build condensed facts from clusters
        facts = []
        for cluster in clusters[:max_facts]:
            fact = self._build_fact(cluster)
            if fact:
                facts.append(fact)

        return facts

    def _filter_noise(
        self,
        messages   : list[dict],
        min_length : int,
    ) -> list[dict]:
        """
        Removes noise messages — too short, too generic, no semantic value.

        Noise patterns:
        - Less than min_length words
        - Only stopwords / filler words
        - Purely punctuation or emoji
        """
        NOISE_WORDS = {
            "ok", "okay", "yes", "no", "yeah", "nope", "sure",
            "lol", "haha", "thanks", "thank", "thx", "bye",
            "hello", "hi", "hey", "coucou", "salut", "merci",
            "cool", "nice", "great", "good", "wow", "ah",
            "oui", "non", "bof", "ouais", "voila", "voilà",
            "d accord", "d'accord", "parfait", "super",
        }

        clean = []
        for msg in messages:
            text = msg.get("text", "").strip()
            words = text.lower().split()

            if len(words) < min_length:
                continue

            non_noise = [w for w in words if w not in NOISE_WORDS]
            if len(non_noise) < 2:
                continue

            clean.append(msg)

        return clean

    def _cluster_messages(
        self,
        vecs            : list[tuple[str, np.ndarray, dict]],
        sim_threshold   : float,
    ) -> list[list[tuple[str, np.ndarray, dict]]]:
        """
        Groups semantically similar messages into clusters.
        Uses greedy single-pass clustering (O(n²) — fine for sessions).

        Each cluster = group of messages about the same topic.
        """
        clusters : list[list[tuple]] = []
        assigned  = [False] * len(vecs)

        for i, (text_i, vec_i, msg_i) in enumerate(vecs):
            if assigned[i]:
                continue

            cluster = [(text_i, vec_i, msg_i)]
            assigned[i] = True

            norm_i = np.linalg.norm(vec_i)
            if norm_i == 0:
                clusters.append(cluster)
                continue
            unit_i = vec_i / norm_i

            for j in range(i + 1, len(vecs)):
                if assigned[j]:
                    continue
                text_j, vec_j, msg_j = vecs[j]
                norm_j = np.linalg.norm(vec_j)
                if norm_j == 0:
                    continue
                sim = float(np.dot(unit_i, vec_j / norm_j))
                if sim >= sim_threshold:
                    cluster.append((text_j, vec_j, msg_j))
                    assigned[j] = True

            clusters.append(cluster)

        # Sort clusters by size descending (most repeated topic first)
        clusters.sort(key=lambda c: len(c), reverse=True)
        return clusters

    def _build_fact(
        self,
        cluster: list[tuple[str, np.ndarray, dict]],
    ) -> dict | None:
        """
        Builds a condensed memory fact from a cluster of similar messages.

        Strategy:
        - Pick the longest/most informative message as the representative
        - Count cluster size as reinforcement signal
        - Infer category from keywords
        """
        if not cluster:
            return None

        # Pick most informative message (longest)
        texts = [text for text, _, _ in cluster]
        representative = max(texts, key=lambda t: len(t.split()))

        reinforcements = len(cluster)
        category = self._infer_category(representative)

        return {
            "action": "summarized_fact",
            "metadata": {
                "text"          : representative,
                "source_count"  : reinforcements,
                "category"      : category,
                "reinforcements": reinforcements,
                "session_summary": True,
            }
        }

    def _infer_category(self, text: str) -> str:
        """
        Simple keyword-based category inference.
        In production, replace with a proper classifier.
        """
        text_lower = text.lower()

        CATEGORIES = {
            "tech_stack"  : ["react", "vue", "python", "javascript", "typescript",
                             "node", "code", "framework", "library", "api",
                             "backend", "frontend", "database", "sql", "git"],
            "preference"  : ["prefer", "like", "love", "hate", "always", "never",
                             "favorite", "dark mode", "light mode", "ui", "ux"],
            "learning"    : ["learning", "studying", "course", "tutorial",
                             "beginner", "understand", "practice"],
            "product"     : ["pricing", "plan", "pro", "upgrade", "subscription",
                             "billing", "free", "trial", "dashboard"],
            "sentiment"   : ["happy", "sad", "excited", "worried", "frustrated",
                             "great", "terrible", "love", "hate"],
            "work"        : ["work", "job", "company", "team", "project",
                             "meeting", "office", "remote", "client"],
        }

        for category, keywords in CATEGORIES.items():
            if any(kw in text_lower for kw in keywords):
                return category

        return "general"

    def recall_chain(
        self,
        seg       ,
        encoder   ,
        user_id   : str,
        seed_vec_id: int,
        depth     : int | None = None,
        now       : float | None = None,
    ) -> list[RecalledMemory]:
        """
        Graph traversal recall starting from a seed memory.
        Finds memories associated with the seed by similarity.
        """
        now   = now or time.time()
        depth = depth or self.config.chain_depth

        idx = seg._indexes.get(user_id)
        if idx is None or seed_vec_id not in idx.nodes:
            return []

        visited  = {seed_vec_id}
        frontier = [seed_vec_id]
        results  = []

        for hop in range(depth):
            next_frontier = []
            for vid in frontier:
                seed_vec = idx.nodes[vid].vector
                neighbors = idx.nodes[vid].neighbors.get(0, [])
                for nid in neighbors:
                    if nid in visited:
                        continue
                    visited.add(nid)
                    if nid not in idx.nodes:
                        continue

                    neighbor_vec = idx.nodes[nid].vector
                    sim = float(np.dot(seed_vec, neighbor_vec))

                    if sim >= self.config.chain_sim_threshold:
                        m = self._build_memory(
                            seg, user_id, nid,
                            cosine_sim=sim,
                            decay_weight=self._decay(
                                seg._metadata[user_id][nid].get("metadata", {}).get("timestamp", now),
                                now
                            ),
                            freq_weight=1.0,
                            recency_weight=1.0,
                            chain_depth=hop + 1,
                            now=now,
                        )
                        if m:
                            results.append(m)
                            next_frontier.append(nid)

            frontier = next_frontier
            if not frontier:
                break

        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:self.config.top_k]

    # ------------------------------------------------------------------
    # Scoring strategies
    # ------------------------------------------------------------------

    def _score_semantic(self, candidates, query_vec, seg, user_id, now):
        """Pure cosine similarity — no temporal or frequency signals."""
        results = []
        for vid, meta in candidates:
            idx = seg._indexes.get(user_id)
            if not idx or vid not in idx.nodes:
                continue
            cos = float(np.dot(query_vec, idx.nodes[vid].vector))
            m = self._build_memory(
                seg, user_id, vid,
                cosine_sim=cos, decay_weight=1.0,
                freq_weight=1.0, recency_weight=1.0,
                chain_depth=0, now=now
            )
            if m:
                results.append(m)
        return results

    def _score_temporal(self, candidates, query_vec, seg, user_id, now):
        """Cosine × decay — no frequency signal."""
        results = []
        for vid, meta in candidates:
            idx = seg._indexes.get(user_id)
            if not idx or vid not in idx.nodes:
                continue
            cos = float(np.dot(query_vec, idx.nodes[vid].vector))
            ts  = meta.get("timestamp", now)
            dw  = self._decay(ts, now)
            m = self._build_memory(
                seg, user_id, vid,
                cosine_sim=cos, decay_weight=dw,
                freq_weight=1.0, recency_weight=1.0,
                chain_depth=0, now=now
            )
            if m:
                results.append(m)
        return results

    def _score_contextual(self, candidates, query_vec, seg, user_id, now):
        """Full multi-signal: cosine × decay × frequency × recency_of_recall."""
        results = []
        for vid, meta in candidates:
            idx = seg._indexes.get(user_id)
            if not idx or vid not in idx.nodes:
                continue

            cos  = float(np.dot(query_vec, idx.nodes[vid].vector))
            ts   = meta.get("timestamp", now)
            dw   = self._decay(ts, now)
            fw   = self._freq_boost(meta.get("reinforcements", 1))
            rw   = self._recency_of_recall_boost(user_id, vid, now)

            m = self._build_memory(
                seg, user_id, vid,
                cosine_sim=cos, decay_weight=dw,
                freq_weight=fw, recency_weight=rw,
                chain_depth=0, now=now
            )
            if m:
                results.append(m)
        return results

    def _score_chain(self, candidates, query_vec, seg, user_id, now):
        """Contextual scoring + chain expansion for top results."""
        base = self._score_contextual(candidates, query_vec, seg, user_id, now)
        base.sort(key=lambda x: x.final_score, reverse=True)

        # Expand top-3 with chain recall
        chain_results = []
        for m in base[:3]:
            chain = self.recall_chain(seg, None, user_id, m.vec_id, depth=1, now=now)
            chain_results.extend(chain)

        # Merge, deduplicate
        seen = {m.vec_id for m in base}
        for m in chain_results:
            if m.vec_id not in seen:
                base.append(m)
                seen.add(m.vec_id)

        return base

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _get_candidates(
        self,
        seg     ,
        user_id : str,
        filters : RecallFilter,
        now     : float,
    ) -> list[tuple[int, dict]]:
        """Returns (vec_id, inner_meta) for all memories passing filters."""
        if user_id not in seg._metadata:
            return []

        results = []
        for vid, meta in seg._metadata[user_id].items():
            if meta.get("__deleted__", False):
                continue

            inner = meta.get("metadata", {})
            status = inner.get("__status__", "active")

            # Status filter
            if filters.status != "all" and status != filters.status:
                continue

            # Category filter
            if filters.category:
                if inner.get("category") != filters.category:
                    continue

            # Age filter
            if filters.max_age_hours is not None:
                ts = inner.get("timestamp", now)
                age_h = (now - ts) / 3600.0
                if age_h > filters.max_age_hours:
                    continue

            results.append((vid, inner))

        return results

    def _build_memory(
        self,
        seg           ,
        user_id       : str,
        vec_id        : int,
        cosine_sim    : float,
        decay_weight  : float,
        freq_weight   : float,
        recency_weight: float,
        chain_depth   : int,
        now           : float,
    ) -> RecalledMemory | None:
        meta  = seg._metadata.get(user_id, {}).get(vec_id)
        if not meta:
            return None

        inner = meta.get("metadata", {})
        ts    = inner.get("timestamp", now)

        final = cosine_sim * decay_weight * freq_weight * recency_weight
        final = max(0.0, min(1.0, final))

        recall_info  = self._recall_log.get(user_id, {}).get(vec_id, {})

        return RecalledMemory(
            vec_id         = vec_id,
            user_id        = user_id,
            text           = inner.get("text", meta.get("action", "")),
            action         = meta.get("action", ""),
            category       = inner.get("category", ""),
            status         = inner.get("__status__", "active"),
            timestamp      = ts,
            last_recalled  = recall_info.get("last_recalled"),
            reinforcements = inner.get("reinforcements", 1),
            recall_count   = recall_info.get("count", 0),
            final_score    = round(final, 6),
            cosine_sim     = round(cosine_sim, 6),
            decay_weight   = round(decay_weight, 6),
            freq_weight    = round(freq_weight, 6),
            recency_weight = round(recency_weight, 6),
            chain_depth    = chain_depth,
            meta           = meta,
        )

    def _decay(self, timestamp: float, now: float) -> float:
        """Exponential decay with 24h half-life (human memory preset)."""
        delta_h = max(0.0, (now - timestamp) / 3600.0)
        lam = math.log(2) / 24.0  # 24h half-life
        raw = math.exp(-lam * delta_h)
        return max(0.05, min(1.0, raw))

    def _freq_boost(self, n: int) -> float:
        """Log-scaled frequency boost, saturates at 10 reinforcements."""
        n = max(1, n)
        boost = 1.0 + 0.30 * (math.log(1 + n) / math.log(1 + 10))
        return min(boost, 1.30)

    def _recency_of_recall_boost(
        self,
        user_id : str,
        vec_id  : int,
        now     : float,
    ) -> float:
        """
        Boost for memories that were recently recalled.
        A memory recalled 5 minutes ago gets a higher weight
        than one that hasn't been recalled in a week.
        """
        log = self._recall_log.get(user_id, {}).get(vec_id, {})
        last_recalled = log.get("last_recalled")
        if last_recalled is None:
            return 1.0

        hours_since_recall = (now - last_recalled) / 3600.0
        # Exponential boost that fades over 6 hours
        boost = 1.0 + (self.config.recency_boost_factor - 1.0) * math.exp(
            -hours_since_recall / 6.0
        )
        return min(boost, self.config.recency_boost_factor)

    def _update_recall_log(
        self,
        user_id : str,
        vec_id  : int,
        seg     ,
        now     : float,
    ) -> None:
        """Updates recall metadata for a retrieved memory."""
        if user_id not in self._recall_log:
            self._recall_log[user_id] = {}

        log = self._recall_log[user_id].get(vec_id, {"count": 0})
        log["count"]         = log.get("count", 0) + 1
        log["last_recalled"] = now
        self._recall_log[user_id][vec_id] = log

        # Also update in segment metadata
        if user_id in seg._metadata and vec_id in seg._metadata[user_id]:
            inner = seg._metadata[user_id][vec_id].get("metadata", {})
            inner["recall_count"]   = log["count"]
            inner["last_recalled"]  = now
            seg._metadata[user_id][vec_id]["metadata"] = inner

    def __repr__(self) -> str:
        return (
            f"RecallEngine("
            f"mode={self.config.mode}, "
            f"top_k={self.config.top_k}, "
            f"chain_depth={self.config.chain_depth})"
        )


if __name__ == "__main__":
    import sys, shutil
    sys.path.insert(0, ".")

    from core.embeddings.encoder import Encoder
    from core.vectordb.engine import VectorDBEngine

    shutil.rmtree("/tmp/quorex_recall_test", ignore_errors=True)

    SEED = [
        {"action": "viewed pricing page", "metadata": {"text": "pricing"}},
        {"action": "searched pricing plan", "metadata": {"text": "searched pricing"}},
        {"action": "upgraded plan pro billing", "metadata": {"text": "upgraded"}},
        {"action": "write code python backend", "metadata": {"text": "python"}},
        {"action": "use dark mode interface", "metadata": {"text": "dark mode"}},
        {"action": "play guitar music hobby", "metadata": {"text": "guitar"}},
        {"action": "buy groceries market food", "metadata": {"text": "shopping"}},
        {"action": "work remote home office", "metadata": {"text": "remote"}},
        {"action": "read book novel story", "metadata": {"text": "reading"}},
        {"action": "drive car road commute", "metadata": {"text": "driving"}},
    ]

    encoder = Encoder(n_components=8)
    encoder.fit(SEED)

    engine = VectorDBEngine(path="/tmp/quorex_recall_test", dim=8)
    engine.start()

    now = time.time()

    def store(uid, action, text, ts_offset_h=0, category="", reinforcements=1):
        meta = {
            "action": action,
            "metadata": {
                "text": text,
                "timestamp": now - ts_offset_h * 3600,
                "reinforcements": reinforcements,
                "category": category,
                "__status__": "active",
            }
        }
        vec = encoder.encode({"action": action})
        engine.insert(uid, vec, meta)

    store("user_123", "viewed pricing page",         "Checked the pricing page",    ts_offset_h=6,  category="product")
    store("user_123", "searched pricing plan",       "Searched for pricing options", ts_offset_h=5,  category="product", reinforcements=3)
    store("user_123", "upgraded plan pro billing",   "Upgraded to Pro plan!",        ts_offset_h=0.1, category="product")
    store("user_123", "write code python backend",   "Working on Python scripts",    ts_offset_h=24, category="tech")
    store("user_123", "use dark mode interface",     "I always use dark mode",       ts_offset_h=2,  category="preference", reinforcements=5)
    store("user_123", "play guitar music hobby",     "Play guitar on weekends",      ts_offset_h=48, category="hobby")

    recall_engine = RecallEngine(RecallConfig(mode=RecallMode.CONTEXTUAL, top_k=4))
    print(recall_engine)
    print()

    # Test 1 — Contextual recall
    print("=== CONTEXTUAL — query: 'pricing' ===")
    results = recall_engine.recall(engine.segment, encoder, "user_123", "pricing", now=now)
    print(f"{'#':<4} {'Text':<35} {'Score':>7} {'Cos':>6} {'Decay':>7} {'Freq':>6} {'Rec':>6} {'Age':>10}")
    print("─" * 85)
    for i, m in enumerate(results):
        print(f"#{i+1:<3} {m.text[:33]:<35} {m.final_score:>7.3f} {m.cosine_sim:>6.3f} {m.decay_weight:>7.3f} {m.freq_weight:>6.3f} {m.recency_weight:>6.3f} {m.age_str:>10}")

    # Test 2 — Semantic only (no decay)
    print("\n=== SEMANTIC (no decay) — query: 'pricing' ===")
    r2 = RecallEngine(RecallConfig(mode=RecallMode.SEMANTIC, top_k=4))
    results2 = r2.recall(engine.segment, encoder, "user_123", "pricing", now=now)
    for i, m in enumerate(results2):
        print(f"#{i+1:<3} {m.text[:33]:<35} score={m.final_score:.3f}  {m.age_str:>10}")

    # Test 3 — Filter by category
    print("\n=== FILTERED — category='product' ===")
    r3 = RecallEngine(RecallConfig(mode=RecallMode.CONTEXTUAL, top_k=4))
    results3 = r3.recall(
        engine.segment, encoder, "user_123", "pricing",
        filters=RecallFilter(category="product"), now=now
    )
    for i, m in enumerate(results3):
        print(f"#{i+1:<3} [{m.category}] {m.text[:40]}  score={m.final_score:.3f}")

    # Test 4 — Summary
    print("\n=== SUMMARY (LLM injection) ===")
    print(recall_engine.summarize(results, user_id="user_123"))

    engine.stop()