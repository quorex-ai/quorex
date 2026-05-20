import math
import json
import os
from collections import Counter
from tokenizer import Tokenizer


class TFIDFVectorizer:
    def __init__(self):
        self.vocabulary: dict[str, int] = {}   # token -> index
        self.idf: dict[str, float] = {}        # token -> idf score
        self.fitted = False
        self.tokenizer = Tokenizer()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, documents: list[str]) -> None:
        """
        Builds vocabulary and IDF from a list of raw text documents.
        Must be called before transform().
        """
        tokenized = [self.tokenizer.tokenize(doc) for doc in documents]
        self._fit_from_tokens(tokenized)

    def fit_events(self, events: list[dict]) -> None:
        """
        Builds vocabulary and IDF from a list of Quorex event dicts.
        """
        tokenized = [self.tokenizer.tokenize_event(e) for e in events]
        self._fit_from_tokens(tokenized)

    def _fit_from_tokens(self, tokenized_docs: list[list[str]]) -> None:
        n_docs = len(tokenized_docs)

        # Build vocabulary
        all_tokens = set(t for doc in tokenized_docs for t in doc)
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}

        # Compute IDF for each token
        # IDF = log((1 + N) / (1 + df)) + 1  — smoothed to avoid zero division
        doc_freq: dict[str, int] = Counter()
        for doc in tokenized_docs:
            for token in set(doc):
                doc_freq[token] += 1

        self.idf = {
            token: math.log((1 + n_docs) / (1 + doc_freq.get(token, 0))) + 1
            for token in self.vocabulary
        }

        self.fitted = True

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, text: str) -> dict[int, float]:
        """
        Transforms raw text into a sparse TF-IDF vector.
        Returns a dict of {index: score} — only non-zero values.
        """
        if not self.fitted:
            raise RuntimeError("Vectorizer must be fitted before calling transform().")

        tokens = self.tokenizer.tokenize(text)
        return self._compute_tfidf(tokens)

    def transform_event(self, event: dict) -> dict[int, float]:
        """
        Transforms a Quorex event dict into a sparse TF-IDF vector.
        """
        if not self.fitted:
            raise RuntimeError("Vectorizer must be fitted before calling transform().")

        tokens = self.tokenizer.tokenize_event(event)
        return self._compute_tfidf(tokens)

    def _compute_tfidf(self, tokens: list[str]) -> dict[int, float]:
        if not tokens:
            return {}

        # TF = count(token) / total tokens in doc
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
        """
        Converts a sparse vector to a dense list of floats.
        Size = vocabulary size.
        """
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
            }, f)

    def load(self, path: str) -> None:
        with open(path, "r") as f:
            data = json.load(f)
        self.vocabulary = data["vocabulary"]
        self.idf = data["idf"]
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

    print(f"Vocabulary size: {len(v.vocabulary)} tokens")
    print(f"Vocabulary: {list(v.vocabulary.keys())}")

    vec = v.transform_event(events[0])
    print(f"\nSparse vector for 'viewed_pricing':")
    for idx, score in vec.items():
        token = list(v.vocabulary.keys())[idx]
        print(f"  {token}: {score:.4f}")