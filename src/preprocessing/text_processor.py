"""Text preprocessing and chunking for cybersecurity documents.

Provides cleaning, normalization, and chunking of raw document text
into smaller segments suitable for embedding and retrieval.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document with content and metadata (mirrors ingestion.Document)."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TextProcessor:
    """Cleans and chunks text documents for the RAG pipeline.

    Attributes:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Example:
        >>> processor = TextProcessor(chunk_size=512, chunk_overlap=50)
        >>> chunks = processor.chunk_text("A very long document text...")
        >>> print(len(chunks))
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            "TextProcessor initialized: chunk_size=%d, chunk_overlap=%d",
            chunk_size, chunk_overlap,
        )

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Operations:
        - Remove null bytes and control characters (keep newlines, tabs)
        - Normalize whitespace (collapse multiple spaces/blank lines)
        - Strip leading/trailing whitespace

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text string.
        """
        if not text:
            return ""

        # Remove null bytes
        text = text.replace("\x00", " ")

        # Remove control characters except newline (\n), carriage return (\r), tab (\t)
        text = re.sub(r"[^\S\n\r\t]+", " ", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse multiple blank lines into at most two newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Uses a sliding window approach with configurable chunk size
        and overlap. Attempts to break at sentence/line boundaries
        when possible.

        Args:
            text: Text to split into chunks.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        # If text fits in a single chunk, return it directly
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Determine end position for this chunk
            end = min(start + self.chunk_size, text_length)

            # Try to find a natural break point (newline or period+space)
            # within the last 20% of the chunk
            if end < text_length:
                search_start = start + int(self.chunk_size * 0.8)
                chunk_text = text[search_start:end]

                # Prefer breaking at double newline, then single newline, then period
                for delimiter in ["\n\n", "\n", ". "]:
                    last_break = chunk_text.rfind(delimiter)
                    if last_break != -1:
                        end = search_start + last_break + len(delimiter)
                        break

            chunk = text[start:end].strip()
            if chunk:  # Don't add empty chunks
                chunks.append(chunk)

            # Move start forward, accounting for overlap
            step = end - start - self.chunk_overlap
            if step <= 0:
                step = max(1, end - start)  # Prevent infinite loop
            start = start + step

        logger.debug(
            "Chunked text (%d chars) into %d chunks",
            text_length, len(chunks),
        )
        return chunks

    def process_documents(self, documents: list[Document]) -> list[Document]:
        """Clean and chunk a list of documents.

        Each input document is split into multiple chunk documents,
        preserving and augmenting the original metadata.

        Args:
            documents: List of raw Document objects.

        Returns:
            List of chunked Document objects with augmented metadata.
        """
        chunked_docs: list[Document] = []

        for doc in documents:
            cleaned = self.clean_text(doc.content)
            chunks = self.chunk_text(cleaned)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "parent_source": doc.metadata.get("source", "unknown"),
                    "chunk_size_chars": len(chunk),
                }
                chunked_docs.append(
                    Document(content=chunk, metadata=chunk_metadata)
                )

            logger.info(
                "Processed '%s': %d chunks from %d chars",
                doc.metadata.get("filename", "unknown"),
                len(chunks),
                len(cleaned),
            )

        logger.info(
            "Total: %d documents -> %d chunks (chunk_size=%d, overlap=%d)",
            len(documents), len(chunked_docs),
            self.chunk_size, self.chunk_overlap,
        )
        return chunked_docs
