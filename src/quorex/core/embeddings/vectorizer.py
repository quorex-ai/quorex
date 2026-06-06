from __future__ import annotations

import math
import json
import os
from collections import Counter
from .tokenizer import Tokenizer


class TFIDFVectorizer:
    def __init__(self):
        self.vocabulary: dict[str, int] = {}   # token -> index
        self.idf: dict[str, float] = {}        # token -> idf score
        self.fitted = False
        self.tokenizer = Tokenizer()

        # Accumulated corpus stats — required for online updates.
        # IDF is a corpus-level statistic, so a partial_fit needs the
        # historical document frequencies, not just the new batch.
        self._doc_freq: Counter = Counter()
        self._n_docs: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, documents: list[str]) -> None:
        """Builds vocabulary and IDF from a list of raw text documents."""
        tokenized = [self.tokenizer.tokenize(doc) for doc in documents]
        self._reset_corpus()
        self._fit_from_tokens(tokenized)

    def fit_events(self, events: list[dict]) -> None:
        """Builds vocabulary and IDF from a list of Quorex event dicts."""
        tokenized = [self.tokenizer.tokenize_event(e) for e in events]
        self._reset_corpus()
        self._fit_from_tokens(tokenized)

    def partial_fit_events(self, events: list[dict]) -> list[str]:
        """
        Online vocabulary update.

        - Tokenizes the new events
        - Adds any new tokens to the vocabulary (new indices appended at the end)
        - Updates document frequencies + total doc count
        - Recomputes IDF for all tokens
        - Returns the list of brand-new tokens added (in the order their
          indices were assigned), so callers can extend a downstream
          projection matrix with the correct column ordering.

        Note: new tokens get vocabulary indices >= previous vocab_size.
        If you're using this together with a fitted reducer, call
        reducer.extend_vocab_with_tokens(new_tokens) so each new vocab
        index gets a real (hashed) projection column instead of a zero
        column.
        """
        tokenized = [self.tokenizer.tokenize_event(e) for e in events]
        return self._partial_fit_from_tokens(tokenized)

    def _reset_corpus(self) -> None:
        self._doc_freq = Counter()
        self._n_docs = 0

    def _fit_from_tokens(self, tokenized_docs: list[list[str]]) -> None:
        all_tokens = set(t for doc in tokenized_docs for t in doc)
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}

        self._doc_freq = Counter()
        for doc in tokenized_docs:
            for token in set(doc):
                self._doc_freq[token] += 1
        self._n_docs = len(tokenized_docs)

        self._recompute_idf()
        self.fitted = True

    def _partial_fit_from_tokens(self, tokenized_docs: list[list[str]]) -> list[str]:
        new_tokens: list[str] = []

        # Discover new tokens, append them at the end of the vocab.
        all_new_tokens = set(t for doc in tokenized_docs for t in doc)
        for token in sorted(all_new_tokens):
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary)
                new_tokens.append(token)

        # Update document frequencies + count.
        for doc in tokenized_docs:
            for token in set(doc):
                self._doc_freq[token] += 1
        self._n_docs += len(tokenized_docs)

        self._recompute_idf()
        self.fitted = True
        return new_tokens

    def _recompute_idf(self) -> None:
        """IDF = log((1 + N) / (1 + df)) + 1 — smoothed."""
        self.idf = {
            token: math.log((1 + self._n_docs) / (1 + self._doc_freq.get(token, 0))) + 1
            for token in self.vocabulary
        }

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, text: str) -> dict[int, float]:
        if not self.fitted:
            raise RuntimeError("Vectorizer must be fitted before calling transform().")
        tokens = self.tokenizer.tokenize(text)
        return self._compute_tfidf(tokens)

    def transform_event(self, event: dict) -> dict[int, float]:
        if not self.fitted:
            raise RuntimeError("Vectorizer must be fitted before calling transform().")
        tokens = self.tokenizer.tokenize_event(event)
        return self._compute_tfidf(tokens)

    def _compute_tfidf(self, tokens: list[str]) -> dict[int, float]:
        if not tokens:
            return {}

        tf = Counter(tokens)
        total = len(tokens)

        vector: dict[int, float] = {}
        for token, count in tf.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf_score = count / total
                idf_score = self.idf[token]
                vector[idx] = tf_score * idf_score

        return vector

    def to_dense(self, sparse_vector: dict[int, float]) -> list[float]:
        dense = [0.0] * len(self.vocabulary)
        for idx, score in sparse_vector.items():
            dense[idx] = score
        return dense

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "vocabulary": self.vocabulary,
                "idf": self.idf,
                "doc_freq": dict(self._doc_freq),
                "n_docs": self._n_docs,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.vocabulary = data["vocabulary"]
        self.idf = data["idf"]
        self._doc_freq = Counter(data.get("doc_freq", {}))
        self._n_docs = data.get("n_docs", 0)
        self.fitted = True


if __name__ == "__main__":
    events = [
        {"action": "viewed_pricing", "metadata": {"plan": "pro", "source": "dashboard"}},
        {"action": "searched_pricing", "metadata": {"query": "pricing", "source": "web"}},
        {"action": "upgraded_plan", "metadata": {"plan": "pro", "source": "billing"}},
        {"action": "visited_homepage", "metadata": {"source": "organic"}},
    ]

    v = TFIDFVectorizer()
    v.fit_events(events)

    print(f"Vocab after fit: {len(v.vocabulary)} tokens")

    new_events = [
        {"action": "clicked_checkout", "metadata": {"source": "mobile"}},
        {"action": "abandoned_cart", "metadata": {"source": "ios"}},
    ]
    added = v.partial_fit_events(new_events)
    print(f"Vocab after partial_fit: {len(v.vocabulary)} tokens (+{added} new)")
