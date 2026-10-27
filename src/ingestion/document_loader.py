"""Document loader for ingesting cybersecurity evidence files.

Supports: .txt, .log, .json, .csv, .pdf
Returns structured Document objects with content and metadata.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single document with its content and metadata.

    Attributes:
        content: The text content of the document.
        metadata: Dictionary containing source information, file type,
                  timestamps, and any other relevant metadata.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Loads cybersecurity evidence files into Document objects.

    Supports multiple file formats commonly used in cybersecurity:
    - .txt, .log: Plain text files (incident reports, text logs)
    - .json: Structured data (Sysmon logs, SIEM alerts)
    - .csv: Tabular data (firewall logs, network logs)
    - .pdf: Document files (threat reports, advisories)

    Example:
        >>> loader = DocumentLoader()
        >>> docs = loader.load_directory("data/sample")
        >>> print(len(docs))
        3
    """

    # File extensions and their loader methods
    SUPPORTED_EXTENSIONS: dict[str, str] = {
        ".txt": "_load_text",
        ".log": "_load_text",
        ".json": "_load_json",
        ".jsonl": "_load_json",
        ".ndjson": "_load_json",
        ".csv": "_load_csv",
        ".pdf": "_load_pdf",
    }

    def load_directory(self, directory: str | Path) -> list[Document]:
        """Load all supported files from a directory.

        Args:
            directory: Path to the directory containing evidence files.

        Returns:
            List of Document objects loaded from all supported files.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        documents: list[Document] = []
        files_found = 0

        for file_path in sorted(directory.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files_found += 1
                try:
                    doc = self.load_file(file_path)
                    documents.append(doc)
                    logger.info("Loaded: %s (%d chars)", file_path.name, len(doc.content))
                except Exception as e:
                    logger.error("Failed to load %s: %s", file_path, e)

        logger.info(
            "Loaded %d documents from %d files in %s",
            len(documents), files_found, directory,
        )
        return documents

    def load_file(self, file_path: str | Path) -> Document:
        """Load a single file based on its extension.

        Args:
            file_path: Path to the file to load.

        Returns:
            Document object with content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        loader_method = getattr(self, self.SUPPORTED_EXTENSIONS[ext])
        return loader_method(file_path)

    def _load_text(self, file_path: Path) -> Document:
        """Load a plain text or log file."""
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return Document(
            content=content,
            metadata=self._base_metadata(file_path, "text"),
        )

    def _load_json(self, file_path: Path) -> Document:
        """Load a JSON or NDJSON/JSONL file and convert to readable text.

        For arrays of objects (e.g., Sysmon logs), each object is
        converted to a text block for chunking.
        """
        data = None
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                parsed_records = []
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        try:
                            parsed_records.append(json.loads(line_str))
                        except json.JSONDecodeError:
                            continue
                if parsed_records:
                    data = parsed_records
                else:
                    raise

        if isinstance(data, list):
            # Array of events/objects — convert each to text block
            text_blocks = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    block = self._dict_to_text(item, record_index=i)
                else:
                    block = str(item)
                text_blocks.append(block)
            content = "\n\n".join(text_blocks)
        elif isinstance(data, dict):
            content = self._dict_to_text(data)
        else:
            content = str(data)

        metadata = self._base_metadata(file_path, "json")
        if isinstance(data, list):
            metadata["record_count"] = len(data)

        return Document(content=content, metadata=metadata)

    def _load_csv(self, file_path: Path) -> Document:
        """Load a CSV file and convert rows to readable text."""
        rows: list[str] = []
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                row_text = " | ".join(
                    f"{key}: {value}" for key, value in row.items() if value
                )
                rows.append(f"Record {i}: {row_text}")

        content = "\n".join(rows)
        metadata = self._base_metadata(file_path, "csv")
        metadata["record_count"] = len(rows)

        return Document(content=content, metadata=metadata)

    def _load_pdf(self, file_path: Path) -> Document:
        """Load a PDF file using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF loading. "
                "Install it with: pip install PyMuPDF"
            )

        doc = fitz.open(str(file_path))
        pages: list[str] = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append(f"--- Page {page_num + 1} ---\n{text}")
        doc.close()

        content = "\n\n".join(pages)
        metadata = self._base_metadata(file_path, "pdf")
        metadata["page_count"] = len(pages)

        return Document(content=content, metadata=metadata)

    @staticmethod
    def _dict_to_text(d: dict[str, Any], record_index: int | None = None) -> str:
        """Convert a dictionary to a human-readable text block."""
        prefix = f"Record {record_index}:\n" if record_index is not None else ""
        lines = [f"  {key}: {value}" for key, value in d.items()]
        return prefix + "\n".join(lines)

    @staticmethod
    def _base_metadata(file_path: Path, file_type: str) -> dict[str, Any]:
        """Create base metadata for a loaded document."""
        return {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": file_type,
            "file_size_bytes": file_path.stat().st_size,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }
