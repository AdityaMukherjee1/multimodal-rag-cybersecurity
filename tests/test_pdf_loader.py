"""Unit tests for PDFLoader with page-level provenance."""

from pathlib import Path
import pytest

from src.ingestion.pdf_loader import PDFLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPDFLoader:
    """Test suite for PDF evidence ingestion."""

    def test_load_cisa_advisory_pdf(self):
        pdf_file = PROJECT_ROOT / "data" / "raw" / "documents" / "pdf" / "cisa_advisory_aa23_347a_briefing.pdf"
        assert pdf_file.exists(), "CISA advisory PDF must exist"

        loader = PDFLoader()
        records = loader.load_file(pdf_file)

        assert len(records) == 1
        rec = records[0]
        assert rec.modality == "pdf"
        assert rec.page_number == 1
        assert rec.metadata["page_number"] == 1
        assert "CYBERSECURITY ADVISORY" in rec.content
        assert rec.raw_reference == str(pdf_file)

    def test_load_nist_pdf_skips_withdrawal_notice(self):
        pdf_file = PROJECT_ROOT / "data" / "raw" / "documents" / "pdf" / "NIST.SP.800-61r2.pdf"
        assert pdf_file.exists(), "NIST SP 800-61r2 PDF must exist"

        loader = PDFLoader(skip_withdrawn_notices=True, max_pages=5)
        records = loader.load_file(pdf_file)

        assert len(records) > 0
        first_rec = records[0]
        # Verify page 1 withdrawal notice is skipped, so first record is page 2
        assert first_rec.page_number == 2
        assert "Computer Security" in first_rec.content
        assert "Withdrawn" not in first_rec.content

    def test_nonexistent_pdf_raises_error(self):
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("nonexistent.pdf")
