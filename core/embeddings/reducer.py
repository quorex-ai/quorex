import json
import os
import numpy as np

class SVDReducer:
    def __init__(self, n_components: int = 64):
        """
        n_components: target number of dimensions after reduction
        64 is a good default for event-based embeddings.
        256 for richer semantic content.
        """
        self.n_components = n_components
        self.components: np.ndarray | None = None # shape: (n_components, vocab_size)
        self.singular_values: np.ndarray | None = None
        self.fitted = False

    # FIT

    def fit(self, matrix: np.ndarray) -> None:
        """
        Fits SVD on a TF-IDF matrix.
        matrix shape: (n_documents, vocab_size)
        """
        if matrix.shape[0] < self.n_components:
            self.n_components = matrix.shape[0]

        # Power iteration SVD - no scipy, pure numpy
        U, s, Vt = self._truncated_svd(matrix, self.n_components)

        self.components = Vt            # (n_components, vocab_size)
        self.singular_values = s        # (n_components)
        self.fitted = True

    def _truncated_svd(
            self, matrix: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ 
        Truncated SVD via randomized power iteration.
        Much fasted than full SVD for sparse/large matrices.
        """
        n, m = matrix.shape 
        rng = np.random.default_rng(42)

        # Random projection
        Omega = rng.standard_normal((m, k))
        Y = matrix @ Omega

        # Power iterations for better accuracy
        for _ in range(3):
            Y = matrix @ (matrix.T @ Y)

        # QR decomposition
        Q, _ = np.linalg.qr(Y)

        # Project matrix
        B = Q.T @ matrix

        # SVD of small matrix B
        U_hat, s, Vt = np.linalg.svd(B, full_matrices=False)

        # Recover U
        U = Q @ U_hat

        return U[:, :k], s[:k], Vt[:k, :]

    # TRANSFORM
    def transform(self, sparse_vector: dict[int, float], vocab_size: int) -> np.ndarray:
        """
        Projects a sparse TF-IDF vector into the reduced space.
        Returns a dense numpy array of shape (n_components,).
        """
        if not self.fitted:
            raise RuntimeError("Reducer must be fitted before calling transform()")
        
        dense = np.zeros(vocab_size)
        for idx, score in sparse_vector.items():
            dense[idx] = score

        reduced = self.components @ dense
        return self._normalize(reduced)
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalization - makes cosine similarity = dot product."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def fit_transform(
            self, vectors: list[dict[int, float]], vocab_size: int 
    ) -> np.ndarray:
        """
        Fits SVD on a list of sparse vectors and returns the reduced dense matrix.
        Shortcuts for fit() + transform() on a batch.
        """
        matrix = np.zeros((len(vectors), vocab_size))
        for i, vec in enumerate(vectors):
            for idx, score in vec.items():
                matrix[i, idx] = score

        self.fit(matrix)

        reduced = matrix @ self.components.T
        norms = np.linalg.norm(reduced, axis=1, keepdims = True)
        norms[norms == 0] = 1
        return reduced / norms

    # PERSISTENCE
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 components=self.components,
                 singular_values=self.singular_values,
                 n_components=np.array([self.n_components]))
        
    def load(self, path: str) -> None:
        data = np.load(path + ".npz")
        self.components = data["components"]
        self.singular_values = data["singular_values"]
        self.n_components = int(data["n_components"][0])
        self.fitted = True

if __name__ == "__main__":
    from vectorizer import TFIDFVectorizer

    events = [
        {"action": "viewed_pricing", "metadata": {"plan": "pro", "source": "dashboard"}},
        {"action": "searched_pricing", "metadata": {"query": "pricing", "source": "web"}},
        {"action": "upgraded_plan", "metadata": {"plan": "pro", "source": "billing"}},
        {"action": "visited_homepage", "metadata": {"source": "organic"}},
        {"action": "viewed_pricing", "metadata": {"plan": "starter", "source": "email"}},
        {"action": "clicked_cta", "metadata": {"source": "dashboard", "plan": "pro"}},
    ]

    # Step 1 : vectorize
    v = TFIDFVectorizer()
    v.fit_events(events)
    sparse_vectors = [v.transform_event(e) for e in events]

    print(f"Vocabulary size: {len(v.vocabulary)}")
    print(f"Sparse vector dims: {len(v.vocabulary)}")

    # Step 2 — reduce
    r = SVDReducer(n_components=4)
    dense_matrix = r.fit_transform(sparse_vectors, len(v.vocabulary))
 
    print(f"\nReduced to: {dense_matrix.shape[1]} dimensions")
    print(f"\nEmbedding for 'viewed_pricing':")
    print(f"  {dense_matrix[0].round(4)}")
    print(f"\nEmbedding for 'searched_pricing':")
    print(f"  {dense_matrix[1].round(4)}")
 
    # Similarity check
    dot = np.dot(dense_matrix[0], dense_matrix[1])
    print(f"\nSimilarity viewed_pricing vs searched_pricing: {dot:.4f}")
    dot2 = np.dot(dense_matrix[0], dense_matrix[3])
    print(f"Similarity viewed_pricing vs visited_homepage:  {dot2:.4f}")