"""Smoke tests for the end-to-end RAG pipeline.

These tests verify that the pipeline components work together
correctly using sample data and the PlaceholderLLM.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.config.settings import Settings
from src.ingestion.document_loader import DocumentLoader
from src.pipeline.rag_pipeline import RAGPipeline


SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


class TestDocumentLoader:
    """Tests for the document loader."""

    def test_load_sample_directory(self):
        """Verify that sample data can be loaded."""
        loader = DocumentLoader()
        docs = loader.load_directory(SAMPLE_DATA_DIR)
        assert len(docs) >= 3  # json, txt, csv
        for doc in docs:
            assert doc.content  # Non-empty content
            assert "source" in doc.metadata
            assert "filename" in doc.metadata

    def test_load_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError):
            DocumentLoader().load_directory("nonexistent_dir")


class TestRAGPipeline:
    """End-to-end smoke tests for the RAG pipeline."""

    @pytest.fixture
    def pipeline_settings(self):
        """Create test settings."""
        return Settings(
            llm_provider="placeholder",
            faiss_index_path="data/processed/test_faiss_index",
        )

    def test_build_index(self, pipeline_settings):
        """Test that the index can be built from sample data."""
        pipeline = RAGPipeline(pipeline_settings)
        summary = pipeline.build_index(SAMPLE_DATA_DIR)

        assert summary["raw_documents"] >= 3
        assert summary["chunks"] > 0
        assert summary["embedding_model"] == pipeline_settings.embedding_model_name
        assert summary["embedding_dimension"] > 0

    def test_query_pipeline(self, pipeline_settings):
        """Test the full query pipeline."""
        pipeline = RAGPipeline(pipeline_settings)
        pipeline.build_index(SAMPLE_DATA_DIR)

        result = pipeline.query("Was suspicious PowerShell activity detected?")

        # Verify response structure
        assert "answer" in result
        assert "evidence" in result
        assert "query_metadata" in result

        # Verify answer is non-empty
        assert len(result["answer"]) > 0

        # Verify evidence structure
        assert len(result["evidence"]) > 0
        for ev in result["evidence"]:
            assert "content" in ev
            assert "source" in ev
            assert "score" in ev
            assert "rank" in ev

        # Verify metadata
        meta = result["query_metadata"]
        assert meta["question"] == "Was suspicious PowerShell activity detected?"
        assert meta["num_results"] > 0
        assert "latency_ms" in meta
        assert "embedding_model" in meta

    def test_query_returns_relevant_results(self, pipeline_settings):
        """Test that PowerShell-related queries retrieve PowerShell evidence."""
        pipeline = RAGPipeline(pipeline_settings)
        pipeline.build_index(SAMPLE_DATA_DIR)

        result = pipeline.query("PowerShell execution bypass")

        # At least one evidence chunk should mention PowerShell
        evidence_text = " ".join(
            ev["content"].lower() for ev in result["evidence"]
        )
        assert "powershell" in evidence_text

    def test_query_without_index_raises_error(self, pipeline_settings):
        """Test that querying without an index raises RuntimeError."""
        pipeline = RAGPipeline(pipeline_settings)
        with pytest.raises(RuntimeError, match="No index loaded"):
            pipeline.query("test question")
