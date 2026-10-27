"""RAG pipeline orchestrating ingestion, processing, retrieval, and generation.

Wires together all pipeline components:
    DocumentLoader -> TextProcessor -> EmbeddingService -> VectorStore -> LLM

Designed for research reproducibility: logs all parameters, retrieval
results, and generated responses.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import Settings
from src.embeddings.embedding_service import EmbeddingService
from src.ingestion.document_loader import Document, DocumentLoader
from src.llm.llm_client import create_llm
from src.preprocessing.text_processor import TextProcessor
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for cybersecurity investigation.

    Orchestrates:
    1. Document ingestion (load files from disk)
    2. Text preprocessing (clean + chunk)
    3. Embedding generation (encode chunks)
    4. Vector storage (FAISS index)
    5. Retrieval (similarity search)
    6. Response generation (LLM or placeholder)

    All parameters are configurable via Settings for experiment
    reproducibility.

    Example:
        >>> from src.config.settings import Settings
        >>> settings = Settings()
        >>> pipeline = RAGPipeline(settings)
        >>> pipeline.build_index("data/sample")
        >>> result = pipeline.query("Was PowerShell executed?")
        >>> print(result["answer"])
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the pipeline with the given settings.

        Args:
            settings: Pipeline configuration. Uses defaults if None.
        """
        self.settings = settings or Settings()

        # Initialize components
        self.loader = DocumentLoader()
        self.processor = TextProcessor(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.embedding_service = EmbeddingService(
            model_name=self.settings.embedding_model_name,
        )
        self.vector_store: VectorStore | None = None
        self.llm = create_llm(
            provider=self.settings.llm_provider,
            model_name=self.settings.llm_model_name,
            api_key=self.settings.llm_api_key,
        )

        # Track processed documents for logging
        self._chunked_documents: list[Document] = []

        logger.info(
            "RAGPipeline initialized with settings: %s",
            json.dumps(self.settings.summary(), indent=2),
        )

    def build_index(
        self,
        data_directory: str | Path,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Build the FAISS index from documents in a directory.

        Pipeline: load -> preprocess -> embed -> index.

        Args:
            data_directory: Path to directory containing evidence files.
            output_path: Optional destination directory for the FAISS index.
                         Defaults to settings.faiss_index_path.

        Returns:
            Summary dict with statistics about the indexing process.
        """
        data_directory = self.settings.resolve_path(str(data_directory))
        logger.info("Building index from: %s", data_directory)

        # 1. Load documents
        raw_documents = self.loader.load_directory(data_directory)
        if not raw_documents:
            raise ValueError(f"No supported documents found in {data_directory}")

        logger.info("Loaded %d raw documents", len(raw_documents))

        # 2. Preprocess and chunk
        # Convert ingestion Documents to preprocessing-compatible format
        self._chunked_documents = self.processor.process_documents(raw_documents)
        logger.info("Created %d chunks", len(self._chunked_documents))

        # 3. Generate embeddings
        chunk_texts = [doc.content for doc in self._chunked_documents]
        embeddings = self.embedding_service.embed_texts(chunk_texts)
        logger.info("Generated embeddings: shape=%s", embeddings.shape)

        # 4. Build FAISS index
        self.vector_store = VectorStore(dimension=embeddings.shape[1])
        self.vector_store.add_documents(embeddings, self._chunked_documents)

        # 5. Save index
        if output_path is not None:
            index_path = self.settings.resolve_path(str(output_path))
        else:
            index_path = self.settings.resolve_path(self.settings.faiss_index_path)
        self.vector_store.save(index_path)
        logger.info("Index saved to: %s", index_path)

        summary = {
            "raw_documents": len(raw_documents),
            "chunks": len(self._chunked_documents),
            "embedding_model": self.settings.embedding_model_name,
            "embedding_dimension": embeddings.shape[1],
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "index_path": str(index_path),
        }
        logger.info("Index build summary: %s", json.dumps(summary, indent=2))
        return summary

    def load_index(self, index_path: str | Path | None = None) -> None:
        """Load a previously saved FAISS index.

        Args:
            index_path: Path to the saved index directory.
                        Defaults to settings.faiss_index_path.
        """
        if index_path is None:
            index_path = self.settings.resolve_path(self.settings.faiss_index_path)
        else:
            index_path = Path(index_path)

        self.vector_store = VectorStore.load(
            index_path,
            dimension=self.settings.embedding_dimension,
        )
        logger.info(
            "Loaded index from %s (%d documents)",
            index_path, len(self.vector_store),
        )

    def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Query the pipeline with a cybersecurity investigation question.

        Pipeline: embed query -> retrieve -> generate response.

        Args:
            question: Natural language question about the evidence.
            top_k: Number of evidence chunks to retrieve.
                   Defaults to settings.top_k.

        Returns:
            Dictionary containing:
            - answer: Generated response text
            - evidence: List of retrieved evidence chunks with metadata
            - query_metadata: Query parameters and timing info
        """
        if self.vector_store is None:
            raise RuntimeError(
                "No index loaded. Call build_index() or load_index() first."
            )

        if top_k is None:
            top_k = self.settings.top_k

        start_time = datetime.now(timezone.utc)

        # 1. Embed the query
        query_embedding = self.embedding_service.embed_query(question)

        # 2. Retrieve relevant chunks
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # 3. Format context for LLM
        context = [
            {
                "content": r.content,
                "metadata": r.metadata,
                "score": r.score,
                "rank": r.rank,
            }
            for r in results
        ]

        # 4. Generate response
        answer = self.llm.generate(prompt=question, context=context)

        end_time = datetime.now(timezone.utc)
        latency_ms = (end_time - start_time).total_seconds() * 1000

        # 5. Build response with full metadata for research logging
        response = {
            "answer": answer,
            "evidence": [
                {
                    "content": r.content,
                    "source": r.metadata.get("filename", "unknown"),
                    "chunk_index": r.metadata.get("chunk_index", -1),
                    "score": r.score,
                    "rank": r.rank,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "query_metadata": {
                "question": question,
                "top_k": top_k,
                "num_results": len(results),
                "embedding_model": self.settings.embedding_model_name,
                "llm_provider": self.settings.llm_provider,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "latency_ms": round(latency_ms, 2),
                "timestamp": end_time.isoformat(),
            },
        }

        logger.info(
            "Query completed: '%s' -> %d results in %.0fms",
            question[:50], len(results), latency_ms,
        )
        return response
