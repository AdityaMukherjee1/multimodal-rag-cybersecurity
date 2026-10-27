"""FAISS-based vector store for document retrieval.

Stores document embeddings and metadata, supporting similarity
search for the RAG pipeline.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result from vector search.

    Attributes:
        content: The text content of the retrieved chunk.
        metadata: Source metadata (filename, chunk index, etc.).
        score: Similarity score (lower = more similar for L2).
        rank: Rank position in the results (1-based).
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0


class VectorStore:
    """FAISS-backed vector store for document embeddings.

    Uses a flat inner-product index (IndexFlatIP) for exact
    nearest-neighbor search. Embeddings should be L2-normalized
    before insertion so that inner product equals cosine similarity.

    Attributes:
        dimension: Embedding vector dimension.

    Example:
        >>> store = VectorStore(dimension=384)
        >>> store.add_documents(embeddings_array, documents_list)
        >>> results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, dimension: int) -> None:
        """Initialize the vector store.

        Args:
            dimension: Dimensionality of the embedding vectors.
        """
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with L2-norm)
        self._documents: list[dict[str, Any]] = []  # Parallel list of {content, metadata}
        logger.info("VectorStore initialized: dimension=%d", dimension)

    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: list[Any],
    ) -> None:
        """Add document embeddings and metadata to the store.

        Args:
            embeddings: NumPy array of shape (n, dimension).
            documents: List of Document-like objects with 'content' and 'metadata' attributes.
                       Must have the same length as embeddings.

        Raises:
            ValueError: If embeddings and documents lengths don't match.
        """
        if len(embeddings) != len(documents):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(documents)} documents"
            )

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.dimension}), "
                f"got {embeddings.shape}"
            )

        # Add to FAISS index
        self._index.add(embeddings.astype(np.float32))

        # Store document data in parallel list
        for doc in documents:
            if isinstance(doc, dict):
                c = doc.get("content", str(doc))
                m = doc.get("metadata", {})
            elif hasattr(doc, "content"):
                c = doc.content
                m = getattr(doc, "metadata", {})
            else:
                c = str(doc)
                m = {}
            self._documents.append({
                "content": c,
                "metadata": m,
            })

        logger.info(
            "Added %d documents to index (total: %d)",
            len(documents), len(self._documents),
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Search for the most similar documents.

        Args:
            query_embedding: Query vector of shape (1, dimension).
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult objects, ranked by similarity.
        """
        if len(self._documents) == 0:
            logger.warning("Search called on empty index.")
            return []

        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query vector dimension {self.dimension}, got {query_embedding.shape[1]}"
            )

        # Clamp top_k to available documents
        effective_k = min(top_k, len(self._documents))

        scores, indices = self._index.search(
            query_embedding.astype(np.float32), effective_k
        )

        results: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            doc_data = self._documents[idx]
            results.append(
                RetrievalResult(
                    content=doc_data["content"],
                    metadata=doc_data["metadata"],
                    score=float(score),
                    rank=rank,
                )
            )

        logger.info(
            "Search returned %d results (top_k=%d)",
            len(results), top_k,
        )
        return results

    def save(self, directory: str | Path) -> None:
        """Save the FAISS index and document metadata to disk.

        Args:
            directory: Directory path to save the index files.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = directory / "index.faiss"
        faiss.write_index(self._index, str(index_path))

        # Save document metadata
        metadata_path = directory / "documents.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(self._documents, f)

        logger.info(
            "Saved index (%d vectors) to %s",
            self._index.ntotal, directory,
        )

    @classmethod
    def load(cls, directory: str | Path, dimension: int) -> VectorStore:
        """Load a saved FAISS index and document metadata.

        Args:
            directory: Directory containing saved index files.
            dimension: Expected embedding dimension.

        Returns:
            Loaded VectorStore instance.
        """
        directory = Path(directory)

        index_path = directory / "index.faiss"
        metadata_path = directory / "documents.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Document metadata not found: {metadata_path}")

        store = cls(dimension=dimension)
        store._index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as f:
            store._documents = pickle.load(f)

        logger.info(
            "Loaded index: %d vectors, %d documents from %s",
            store._index.ntotal, len(store._documents), directory,
        )
        return store

    def __len__(self) -> int:
        """Return the number of documents in the store."""
        return len(self._documents)
