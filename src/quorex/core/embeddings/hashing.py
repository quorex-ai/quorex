"""
quorex.core.embeddings.hashing
-------------------------------
Dependency-free embedding encoder (NumPy + stdlib only).

Replaces the TF-IDF + SVD pipeline, which collapsed on unseen / non-English
vocabulary and produced a variable output dimension (the cause of the
"cannot reshape" crash).

Approach: the *hashing trick* over character n-grams.

  - Every token is expanded into character n-grams (with word boundaries).
  - Each n-gram is hashed (blake2b, deterministic) into a fixed-size vector
    with a signed bucket (signed hashing reduces collisions).
  - The vector is L2-normalized so cosine == dot product.

Properties that fix the engine:
  * No vocabulary to fit  -> no out-of-vocabulary collapse, any language works.
  * Fixed output dimension -> the engine dim and the encoder dim can never
    disagree, so snapshots never break on a dimension mismatch.
  * Stateless / deterministic -> nothing to persist, nothing to grow online.

It is purely lexical (it does not understand synonyms), but two phrases that
share words share n-grams, which is exactly what conflict detection needs:
"same topic, different value".
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np

_TOKEN_RE = re.compile(r"[0-9a-zàâäéèêëîïôöùûüçñ]+", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def _ngrams(token: str, n_min: int, n_max: int) -> list[str]:
    """Character n-grams of a token, plus the whole word, with boundaries."""
    marked = f"^{token}$"
    grams = [f"#{token}"]  # whole-word feature (distinct namespace)
    L = len(marked)
    for n in range(n_min, n_max + 1):
        if L < n:
            if n == n_min:
                grams.append(marked)
            break
        for i in range(L - n + 1):
            grams.append(marked[i:i + n])
    return grams


def _bucket(gram: str, dim: int) -> tuple[int, float]:
    """Deterministic (index, sign) for an n-gram via blake2b."""
    h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, "big")
    idx = v % dim
    sign = 1.0 if (v // dim) & 1 else -1.0
    return idx, sign


class HashingEncoder:
    """
    Drop-in replacement for the Encoder. Same public surface:
    encode(event), encode_text(text), encode_batch(events), fit/save/load.
    """

    def __init__(self, dim: int = 256, n_min: int = 3, n_max: int = 5):
        self.dim = int(dim)
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.fitted = True
        # Compatibility shims so MemoryManager.stats()/start() keep working.
        self.n_components = self.dim
        self.vectorizer = _VocabShim()
        self.reducer = _ReducerShim(self.dim)

    # -- embedding -----------------------------------------------------------

    def _embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(text):
            for gram in _ngrams(tok, self.n_min, self.n_max):
                idx, sign = _bucket(gram, self.dim)
                vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    @staticmethod
    def _event_text(event) -> str:
        if not isinstance(event, dict):
            return str(event)
        parts: list[str] = []
        if event.get("action"):
            parts.append(str(event["action"]))
        md = event.get("metadata", {})
        if isinstance(md, dict):
            for k, v in md.items():
                if k in ("timestamp", "reinforcements") or str(k).startswith("__"):
                    continue
                parts.append(str(v))
        else:
            parts.append(str(md))
        return " ".join(parts)

    # -- public API ----------------------------------------------------------

    def encode(self, event: dict) -> np.ndarray:
        return self._embed_text(self._event_text(event))

    def encode_text(self, text: str) -> np.ndarray:
        return self._embed_text(text)

    def encode_batch(self, events: list[dict]) -> np.ndarray:
        return np.array([self.encode(e) for e in events])

    # -- lifecycle (stateless: these are essentially no-ops) -----------------

    def fit(self, events: list[dict]) -> None:
        self.fitted = True
        print(f"HashingEncoder ready — dim={self.dim}, char {self.n_min}-{self.n_max}grams (stateless)")

    def partial_fit(self, events: list[dict]) -> int:
        return 0

    def refit(self) -> None:
        return None

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "hashing.json"), "w") as f:
            json.dump({"dim": self.dim, "n_min": self.n_min, "n_max": self.n_max}, f)

    def load(self, directory: str) -> None:
        path = os.path.join(directory, "hashing.json")
        if os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
            self.dim = int(cfg["dim"])
            self.n_min = int(cfg["n_min"])
            self.n_max = int(cfg["n_max"])
            self.n_components = self.dim
            self.reducer = _ReducerShim(self.dim)
        self.fitted = True

    def __repr__(self) -> str:
        return f"HashingEncoder(dim={self.dim}, ngrams={self.n_min}-{self.n_max})"


class _VocabShim:
    """Minimal stand-in for Encoder.vectorizer.vocabulary (used in stats)."""
    vocabulary: dict = {}


class _ReducerShim:
    def __init__(self, n_components: int):
        self.n_components = n_components


if __name__ == "__main__":
    enc = HashingEncoder(dim=256)

    def cos(a, b):
        return float(np.dot(enc.encode_text(a), enc.encode_text(b)))

    pairs = [
        ("Pour mes paiements fournisseurs on utilise Stripe",
         "Finalement on est passes a Dagdapay pour les paiements fournisseurs"),
        ("j'aime le bleu", "j'aime le rouge"),
        ("j'aime le bleu", "j'aime le bleu"),
        ("ma couleur preferee", "j'aime le rouge"),
        ("Pour mes paiements fournisseurs on utilise Stripe",
         "je joue de la guitare le weekend"),
    ]
    for a, b in pairs:
        print(f"cos={cos(a, b):.3f}  |  {a!r}  ~  {b!r}")
