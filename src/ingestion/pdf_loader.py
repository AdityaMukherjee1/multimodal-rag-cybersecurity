"""PDF document evidence loader preserving page-level provenance.

Uses PyMuPDF to extract text from PDF cybersecurity reports and advisories,
preserves 1-indexed page numbers in CanonicalEvidenceRecord objects,
and handles formal publication withdrawal notices (e.g. NIST SP 800-61r2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingestion.evidence_schema import CanonicalEvidenceRecord, ModalityType

logger = logging.getLogger(__name__)


class PDFLoader:
    """Loads PDF documents into page-level CanonicalEvidenceRecord objects."""

    def __init__(
        self,
        skip_withdrawn_notices: bool = True,
        max_pages: int | None = None,
    ) -> None:
        """Initialize the PDF loader.

        Args:
            skip_withdrawn_notices: Whether to skip pages containing archived/withdrawn preamble notices.
            max_pages: Optional page limit.
        """
        self.skip_withdrawn_notices = skip_withdrawn_notices
        self.max_pages = max_pages

    def load_file(
        self,
        file_path: str | Path,
        source_dataset: str = "documents_pdf",
    ) -> list[CanonicalEvidenceRecord]:
        """Load a PDF file and produce CanonicalEvidenceRecords for each page.

        Args:
            file_path: Path to the .pdf file.
            source_dataset: Originating dataset name.

        Returns:
            List of CanonicalEvidenceRecord objects with page-level provenance.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            import pymupdf
        except ImportError:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

        records: list[CanonicalEvidenceRecord] = []
        doc = pymupdf.open(str(file_path))
        total_pages = len(doc)
        logger.info("Opening PDF: %s (%d pages)", file_path.name, total_pages)

        for page_idx in range(total_pages):
            if self.max_pages and len(records) >= self.max_pages:
                break

            page_num = page_idx + 1
            page = doc[page_idx]
            page_text = page.get_text().strip()

            if not page_text:
                continue

            # Check if this page is a withdrawal notice preamble (e.g. NIST SP 800-61r2 cover)
            if self.skip_withdrawn_notices and page_idx == 0:
                lower_text = page_text.lower()
                if "withdrawn nist technical series publication" in lower_text or "warning notice" in lower_text:
                    logger.info("Skipping withdrawal notice on page 1 of %s", file_path.name)
                    continue

            evidence_id = f"PDF-{file_path.stem}-p{page_num:03d}"

            content = (
                f"[PDF DOCUMENT EVIDENCE] Source: {file_path.name} | Page {page_num} of {total_pages}\n"
                f"{page_text}"
            )

            rec = CanonicalEvidenceRecord(
                evidence_id=evidence_id,
                source_file=file_path.name,
                source_dataset=source_dataset,
                modality=ModalityType.PDF.value,
                content=content,
                raw_reference=str(file_path),
                metadata={
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "char_count": len(page_text),
                },
                page_number=page_num,
                event_type="document_page",
                chunk_id=page_idx,
            )
            records.append(rec)

        doc.close()
        logger.info("Extracted %d canonical page records from %s", len(records), file_path.name)
        return records
