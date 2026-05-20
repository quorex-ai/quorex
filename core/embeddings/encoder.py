import numpy as np
import os
from tokenizer import Tokenizer
from vectorizer import TFIDFVectorizer
from reducer import SVDReducer


class Encoder:
    def __init__(self, n_components: int = 64):
        self.tokenizer = Tokenizer()
        self.vectorizer = TFIDFVectorizer()
        self.reducer = SVDReducer(n_components=n_components)
        self.fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, events: list[dict]) -> None:
        """
        Trains the full pipeline on a list of Quorex events.
        Must be called before encode().

        Steps:
        1. Tokenize all events
        2. Fit TF-IDF vectorizer → build vocabulary + IDF
        3. Transform all events to sparse vectors
        4. Fit SVD reducer → learn projection matrix
        """
        if not events:
            raise ValueError("Cannot fit on empty event list.")

        # Step 1+2 — fit vectorizer
        self.vectorizer.fit_events(events)

        # Step 3 — transform to sparse
        sparse = [self.vectorizer.transform_event(e) for e in events]

        # Step 4 — fit reducer
        self.reducer.fit_transform(sparse, len(self.vectorizer.vocabulary))

        self.fitted = True
        print(f"Encoder fitted on {len(events)} events")
        print(f"  Vocabulary : {len(self.vectorizer.vocabulary)} tokens")
        print(f"  Output dims: {self.reducer.n_components}")

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, event: dict) -> np.ndarray:
        """
        Encodes a single event into a dense embedding vector.
        Returns np.ndarray of shape (n_components,).
        """
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted before calling encode().")

        sparse = self.vectorizer.transform_event(event)
        dense = self.reducer.transform(sparse, len(self.vectorizer.vocabulary))
        return dense

    def encode_batch(self, events: list[dict]) -> np.ndarray:
        """
        Encodes a list of events into a dense matrix.
        Returns np.ndarray of shape (n_events, n_components).
        """
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted before calling encode_batch().")

        return np.array([self.encode(e) for e in events])

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encodes raw text (not an event dict) into a dense vector.
        Useful for query-time retrieval.
        """
        if not self.fitted:
            raise RuntimeError("Encoder must be fitted before calling encode_text().")

        sparse = self.vectorizer.transform(text)
        return self.reducer.transform(sparse, len(self.vectorizer.vocabulary))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir: str) -> None:
        """Saves the full encoder state to a directory."""
        os.makedirs(dir, exist_ok=True)
        self.vectorizer.save(os.path.join(dir, "vectorizer.json"))
        self.reducer.save(os.path.join(dir, "reducer"))
        print(f"Encoder saved to {dir}/")

    def load(self, dir: str) -> None:
        """Loads the full encoder state from a directory."""
        self.vectorizer.load(os.path.join(dir, "vectorizer.json"))
        self.reducer.load(os.path.join(dir, "reducer"))
        self.fitted = True
        print(f"Encoder loaded from {dir}/")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self.fitted else "not fitted"
        return (
            f"Encoder("
            f"vocab={len(self.vectorizer.vocabulary)}, "
            f"dims={self.reducer.n_components}, "
            f"status={status})"
        )


if __name__ == "__main__":
    from similarity import SimilarityEngine

    # Training events
    events = [
        {"action": "viewed_pricing", "metadata": {"plan": "pro", "source": "dashboard"}},
        {"action": "searched_pricing", "metadata": {"query": "pricing", "source": "web"}},
        {"action": "upgraded_plan", "metadata": {"plan": "pro", "source": "billing"}},
        {"action": "visited_homepage", "metadata": {"source": "organic"}},
        {"action": "clicked_cta", "metadata": {"source": "dashboard", "plan": "pro"}},
        {"action": "viewed_pricing", "metadata": {"plan": "starter", "source": "email"}},
    ]

    # 1 — Fit
    encoder = Encoder(n_components=4)
    encoder.fit(events)
    print(f"\n{encoder}\n")

    # 2 — Encode all events
    vectors = encoder.encode_batch(events)

    # 3 — Build similarity index
    engine = SimilarityEngine()
    engine.add_batch(vectors, events)

    # 4 — Query
    query = {"action": "viewed_pricing", "metadata": {"plan": "pro", "source": "web"}}
    query_vec = encoder.encode(query)
    results = engine.search(query_vec, top_k=3)

    print("Query:", query)
    print("\nTop 3 results:")
    for r in results:
        print(f"  #{r['rank']} score={r['score']} → {r['meta']['action']}")

    # 5 — Save + reload
    encoder.save("/tmp/quorex_encoder")
    encoder2 = Encoder()
    encoder2.load("/tmp/quorex_encoder")

    vec2 = encoder2.encode(query)
    print(f"\nReloaded encoder — same vector: {np.allclose(query_vec, vec2)}")