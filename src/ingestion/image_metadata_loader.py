"""Image evidence loader for screenshots, topology diagrams, and visual log tables.

Extracts image metadata (dimensions, format, color mode, file size) and structures
the evidence record with visual caption and description placeholders, ready for
Phase 3 Vision-Language Model (VLM) integration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.ingestion.evidence_schema import CanonicalEvidenceRecord, ModalityType

logger = logging.getLogger(__name__)


class ImageEvidenceLoader:
    """Loads and inspects image-based cybersecurity evidence."""

    def load_file(
        self,
        file_path: str | Path,
        modality: str = ModalityType.IMAGE_SCREENSHOT.value,
        source_dataset: str = "images",
        description: str | None = None,
        attck_technique: str | None = None,
    ) -> CanonicalEvidenceRecord:
        """Load an image file and create a CanonicalEvidenceRecord.

        Args:
            file_path: Path to the image file (.png, .jpg, etc.).
            modality: Specific image modality (screenshot, diagram, table).
            source_dataset: Dataset name.
            description: Optional textual description or OCR caption of the image.
            attck_technique: Optional ATT&CK technique ID.

        Returns:
            CanonicalEvidenceRecord containing image metadata and visual context.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for image loading. Install with: pip install Pillow")

        with Image.open(str(file_path)) as img:
            width, height = img.size
            img_format = img.format or file_path.suffix.lstrip(".").upper()
            mode = img.mode

        evidence_id = f"IMG-{file_path.stem}"
        visual_desc = description or f"Visual security artifact ({modality}): {file_path.name}"

        content = (
            f"[VISUAL CYBERSECURITY EVIDENCE: {modality.upper()}]\n"
            f"Image File: {file_path.name} ({img_format}, {width}x{height} pixels, Mode: {mode})\n"
            f"Visual Content Description:\n  {visual_desc}\n"
            f"Reference Path: {file_path}"
        )

        metadata: dict[str, Any] = {
            "width": width,
            "height": height,
            "format": img_format,
            "mode": mode,
            "file_size_bytes": file_path.stat().st_size,
            "description": visual_desc,
        }

        logger.info("Loaded image evidence: %s (%dx%d, %s)", file_path.name, width, height, img_format)

        return CanonicalEvidenceRecord(
            evidence_id=evidence_id,
            source_file=file_path.name,
            source_dataset=source_dataset,
            modality=modality,
            content=content,
            raw_reference=str(file_path),
            metadata=metadata,
            event_type="visual_evidence",
            attck_technique=attck_technique,
        )
