"""Embedding service for generating vector representations of text.

Wraps SentenceTransformers to provide a clean interface for the
RAG pipeline. The embedding model is fully configurable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates embeddings using SentenceTransformers models.

    The model is configurable to support research experiments
    comparing different embedding models (BGE-small, BGE-base,
    BGE-large, E5, etc.).

    Attributes:
        model_name: Name/path of the SentenceTransformers model.
        dimension: Dimension of the output embeddings.

    Example:
        >>> service = EmbeddingService("BAAI/bge-small-en-v1.5")
        >>> embeddings = service.embed_texts(["PowerShell executed"])
        >>> print(embeddings.shape)
        (1, 384)
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        """Initialize the embedding service.

        Args:
            model_name: HuggingFace model name or local path.
                        Defaults to BAAI/bge-small-en-v1.5.
        """
        self.model_name = model_name
        self._model: Any = None  # Lazy-loaded
        logger.info("EmbeddingService configured with model: %s", model_name)

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        logger.info("Loading embedding model: %s ...", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        dim = (
            self._model.get_embedding_dimension()
            if hasattr(self._model, "get_embedding_dimension")
            else self._model.get_sentence_embedding_dimension()
        )
        logger.info("Model loaded. Embedding dimension: %d", dim)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension of the loaded model."""
        self._load_model()
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding.

        Returns:
            NumPy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 0)

        self._load_model()

        logger.info("Embedding %d texts (batch_size=%d)...", len(texts), batch_size)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
            convert_to_numpy=True,
        )

        logger.info("Generated embeddings: shape=%s", embeddings.shape)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a single query.

        Args:
            query: Query text string.

        Returns:
            NumPy array of shape (1, dimension).
        """
        if not query:
            raise ValueError("Query text cannot be empty.")

        embedding = self.embed_texts([query])
        return embedding
