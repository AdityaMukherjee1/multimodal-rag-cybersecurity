"""FAISS-based visual vector store for cross-modal image retrieval.

Stores 512-dimensional normalized CLIP visual embeddings for cybersecurity images
(screenshots, diagrams, tables) and supports zero-shot natural language query
similarity search alongside image-to-image search.
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
class VisualRetrievalResult:
    """A single visual retrieval result from cross-modal vector search.

    Attributes:
        evidence_id: Unique evidence identifier (e.g. 'IMG-siem_alert_powershell_execution').
        source_file: Filename of the image artifact.
        raw_reference: File path to the original raw image.
        modality: Modality tag (image_screenshot, image_diagram, image_table).
        content: Canonical text summary containing OCR and VLM description.
        metadata: Image attributes (dimensions, OCR text, VLM caption, etc.).
        score: Cosine similarity score (0.0 to 1.0).
        rank: 1-indexed rank position.
    """

    evidence_id: str
    source_file: str
    raw_reference: str
    modality: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0


class VisualVectorStore:
    """FAISS index dedicated to 512-dimensional CLIP visual embeddings."""

    def __init__(self, dimension: int = 512) -> None:
        """Initialize the visual vector store.

        Args:
            dimension: Dimensionality of CLIP visual embeddings (defaults to 512).
        """
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)  # Cosine similarity for normalized vectors
        self._records: list[dict[str, Any]] = []
        logger.info("VisualVectorStore initialized: dimension=%d", dimension)

    @property
    def total_vectors(self) -> int:
        """Return total number of indexed visual vectors."""
        return self._index.ntotal

    def add_records(
        self,
        embeddings: np.ndarray,
        records: list[dict[str, Any]],
    ) -> None:
        """Add image embeddings and corresponding metadata records.

        Args:
            embeddings: NumPy array of shape (N, dimension).
            records: Parallel list of dictionaries containing image metadata.
        """
        if len(embeddings) != len(records):
            raise ValueError(
                f"Count mismatch: {len(embeddings)} embeddings vs {len(records)} records"
            )

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings of shape (N, {self.dimension}), got {embeddings.shape}"
            )

        self._index.add(embeddings.astype(np.float32))
        self._records.extend(records)
        logger.info("Added %d visual records to index (total: %d)", len(records), self.total_vectors)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[VisualRetrievalResult]:
        """Search for most similar visual evidence artifacts.

        Args:
            query_embedding: Query vector of shape (1, dimension).
            top_k: Maximum number of visual results to return.

        Returns:
            List of VisualRetrievalResult objects ranked by cosine similarity.
        """
        if self.total_vectors == 0:
            logger.warning("VisualVectorStore is empty.")
            return []

        if query_embedding.ndim != 2 or query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query vector of shape (1, {self.dimension}), got {query_embedding.shape}"
            )

        k = min(top_k, self.total_vectors)
        scores, indices = self._index.search(query_embedding.astype(np.float32), k)

        results: list[VisualRetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self._records):
                continue
            rec = self._records[idx]
            results.append(
                VisualRetrievalResult(
                    evidence_id=rec.get("evidence_id", f"IMG-{idx}"),
                    source_file=rec.get("source_file", "unknown_image"),
                    raw_reference=rec.get("raw_reference", ""),
                    modality=rec.get("modality", "image_screenshot"),
                    content=rec.get("content", ""),
                    metadata=rec.get("metadata", {}),
                    score=float(score),
                    rank=rank,
                )
            )

        logger.info("Visual search returned %d results (top_k=%d)", len(results), top_k)
        return results

    def save(self, directory: str | Path) -> None:
        """Persist visual index and record metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "visual_index.faiss"
        metadata_path = directory / "visual_records.pkl"

        faiss.write_index(self._index, str(index_path))
        with open(metadata_path, "wb") as f:
            pickle.dump(self._records, f)

        logger.info("Saved visual index (%d vectors) to %s", self.total_vectors, directory)

    @classmethod
    def load(cls, directory: str | Path, dimension: int = 512) -> VisualVectorStore:
        """Load visual index and record metadata from disk."""
        directory = Path(directory)
        index_path = directory / "visual_index.faiss"
        metadata_path = directory / "visual_records.pkl"

        if not index_path.exists():
            raise FileNotFoundError(f"Visual FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Visual records metadata not found: {metadata_path}")

        store = cls(dimension=dimension)
        store._index = faiss.read_index(str(index_path))
        with open(metadata_path, "rb") as f:
            store._records = pickle.load(f)

        logger.info("Loaded visual index (%d vectors) from %s", store.total_vectors, directory)
        return store
