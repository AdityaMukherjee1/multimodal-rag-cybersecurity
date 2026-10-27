"""Unit tests for the TextProcessor module."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.preprocessing.text_processor import TextProcessor, Document


class TestCleanText:
    """Tests for TextProcessor.clean_text()."""

    def setup_method(self):
        self.processor = TextProcessor(chunk_size=512, chunk_overlap=50)

    def test_empty_string(self):
        assert self.processor.clean_text("") == ""

    def test_removes_null_bytes(self):
        assert self.processor.clean_text("hello\x00world") == "hello world"

    def test_normalizes_whitespace(self):
        result = self.processor.clean_text("hello    world")
        assert "    " not in result
        assert "hello" in result and "world" in result

    def test_collapses_blank_lines(self):
        text = "line1\n\n\n\n\nline2"
        result = self.processor.clean_text(text)
        assert result == "line1\n\nline2"

    def test_strips_whitespace(self):
        result = self.processor.clean_text("   hello   ")
        assert result == "hello"

    def test_normalizes_line_endings(self):
        result = self.processor.clean_text("line1\r\nline2\rline3")
        assert "\r" not in result
        assert "line1\nline2\nline3" == result


class TestChunkText:
    """Tests for TextProcessor.chunk_text()."""

    def test_empty_string(self):
        processor = TextProcessor(chunk_size=100, chunk_overlap=10)
        assert processor.chunk_text("") == []

    def test_short_text_single_chunk(self):
        processor = TextProcessor(chunk_size=100, chunk_overlap=10)
        result = processor.chunk_text("Short text.")
        assert len(result) == 1
        assert result[0] == "Short text."

    def test_long_text_multiple_chunks(self):
        processor = TextProcessor(chunk_size=50, chunk_overlap=10)
        text = "A" * 200
        chunks = processor.chunk_text(text)
        assert len(chunks) > 1

    def test_all_text_covered(self):
        """Verify that all content from the original text appears in at least one chunk."""
        processor = TextProcessor(chunk_size=50, chunk_overlap=10)
        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = processor.chunk_text(text)
        joined = " ".join(chunks)
        # Every word from the original should appear in the chunked output
        for word in ["quick", "brown", "fox", "jumps", "lazy", "dog"]:
            assert word in joined

    def test_chunk_size_respected(self):
        processor = TextProcessor(chunk_size=100, chunk_overlap=10)
        text = "word " * 500
        chunks = processor.chunk_text(text)
        for chunk in chunks:
            # Allow some flexibility due to boundary finding
            assert len(chunk) <= 110  # slight tolerance


class TestChunkValidation:
    """Tests for TextProcessor constructor validation."""

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextProcessor(chunk_size=0, chunk_overlap=10)

    def test_negative_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TextProcessor(chunk_size=100, chunk_overlap=-1)

    def test_overlap_exceeds_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            TextProcessor(chunk_size=100, chunk_overlap=100)


class TestProcessDocuments:
    """Tests for TextProcessor.process_documents()."""

    def test_process_single_document(self):
        processor = TextProcessor(chunk_size=100, chunk_overlap=10)
        docs = [
            Document(
                content="This is a test document with enough text. " * 5,
                metadata={"source": "test.txt", "filename": "test.txt"},
            )
        ]
        result = processor.process_documents(docs)
        assert len(result) >= 1
        assert all(isinstance(d, Document) for d in result)
        assert all("chunk_index" in d.metadata for d in result)
        assert all("parent_source" in d.metadata for d in result)

    def test_metadata_preserved(self):
        processor = TextProcessor(chunk_size=1000, chunk_overlap=50)
        docs = [
            Document(
                content="Short document.",
                metadata={"source": "test.txt", "filename": "test.txt", "custom_field": "value"},
            )
        ]
        result = processor.process_documents(docs)
        assert len(result) == 1
        assert result[0].metadata["custom_field"] == "value"
        assert result[0].metadata["parent_source"] == "test.txt"
