import numpy as np

class SimilarityEngine:
    def __init__(self):
        self.index: np.ndarray | None = None       # (n_stored, n_dims)
        self.metadata: list[dict] = []             # event metadata per vector
    
    # INDEX MANAGEMENT
    def add(self, vector: np.ndarray, meta: dict) -> None:
        """
        Adds a single vector + its metadata to the index.
        """
        vec = vector.reshape(1, -1)

        if self.index is None:
            self.index = vec
        else:
            self.index = np.vstack([self.index, vec])

        self.metadata.append(meta)

    def add_batch(self, vectors: np.ndarray, metas: list[dict]) -> None:
        """
        Adds a batch of vectors + metadata to the index.
        vectors shape: (n, n_dims)
        """
        if self.index is None:
            self.index = vectors
        else:
            self.index = np.vstack([self.index, vectors])

        self.metadata.extend(metas)

    def clear(self) -> None:
        self.index = None
        self.metadata = []

    # SEARCH
    def search(
            self, query: np.ndarray, top_k: int = 5, threshold: float = 0.0
    ) -> list[dict]:
        """ 
        Returns the top_k most similar vectors to the query.

        Since all vectors are L2-normalized (done in reducer),
        consine similarity = dot product.

        Returns list of:
        {
            "score": float,
            "rank": int,
            "meta": dict
        }
        """
        if self.index is None or len(self.metadata) == 0:
            return []
        
        # NORMALIZE QUERY
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # CONSINE SIMILARITY = DOT PRODUCT (VECTORS ARE ALREADY NORMALIZED)
        scores = self.index @ query # (n_stored,)

        # GET TOP_K INDICES SORTED BY SCORE DESCENDING
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score < threshold:
                break
            results.append({
                "rank": rank + 1,
                "score": score,
                "meta": self.metadata[idx],
            })

        return results
    
    def search_batch(
            self, queries: np.ndarray, top_k: int = 5
    ) -> list[list[dict]]:
        """
        Searches for multiple queries at once.
        Returns a list of result list.
        """
        return [self.search(q, top_k) for q in queries]
    

    # STATS
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __repr__(self) -> str:
        dims = self.index.shape[1] if self.index is not None else 0
        return f"SimilarityEngine(stored={len(self)}, dims={dims})"
    
if __name__ == "__main__":
    from vectorizer import TFIDFVectorizer
    from reducer import SVDReducer
 
    events = [
        {"action": "viewed_pricing", "metadata": {"plan": "pro", "source": "dashboard"}},
        {"action": "searched_pricing", "metadata": {"query": "pricing", "source": "web"}},
        {"action": "upgraded_plan", "metadata": {"plan": "pro", "source": "billing"}},
        {"action": "visited_homepage", "metadata": {"source": "organic"}},
        {"action": "clicked_cta", "metadata": {"source": "dashboard", "plan": "pro"}},
        {"action": "viewed_pricing", "metadata": {"plan": "starter", "source": "email"}},
    ]
 
    # Pipeline: vectorize → reduce
    v = TFIDFVectorizer()
    v.fit_events(events)
    sparse = [v.transform_event(e) for e in events]
 
    r = SVDReducer(n_components=4)
    dense = r.fit_transform(sparse, len(v.vocabulary))
 
    # Build index
    engine = SimilarityEngine()
    engine.add_batch(dense, events)
 
    print(engine)
    print()
 
    # Query — find events similar to "viewed_pricing"
    query_event = {"action": "viewed_pricing", "metadata": {"plan": "pro", "source": "web"}}
    query_sparse = v.transform_event(query_event)
    query_dense = r.transform(query_sparse, len(v.vocabulary))
 
    results = engine.search(query_dense, top_k=3)
 
    print("Query: viewed_pricing · pro · web")
    print("Top 3 similar events:\n")
    for r in results:
        print(f"  #{r['rank']} score={r['score']} → {r['meta']}")