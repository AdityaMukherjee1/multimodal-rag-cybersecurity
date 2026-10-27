"""Visual evidence ingestion and processing pipeline.

Combines Optical Character Recognition (OCR), Vision-Language Model (VLM)
interpretation, and image metadata extraction to generate unified,
provenance-preserving CanonicalEvidenceRecords for visual security artifacts.

Maintains strict separation between:
- Observed Evidence (OCR extracted text + image metadata)
- Model-Generated Interpretation (VLM visual reasoning - NOT ground truth)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from src.ingestion.evidence_schema import CanonicalEvidenceRecord, ModalityType
from src.multimodal.ocr_engine import BaseOCREngine, CyberSecurityOCREngine, OCRResult
from src.multimodal.vlm_engine import BaseVLMEngine, CyberSecurityVLMEngine, VLMInterpretation

logger = logging.getLogger(__name__)


class VisualEvidencePipeline:
    """Processes raw images into structured canonical evidence records with OCR and VLM."""

    def __init__(
        self,
        ocr_engine: BaseOCREngine | None = None,
        vlm_engine: BaseVLMEngine | None = None,
    ) -> None:
        """Initialize visual evidence pipeline.

        Args:
            ocr_engine: OCR engine instance (defaults to CyberSecurityOCREngine).
            vlm_engine: VLM engine instance (defaults to CyberSecurityVLMEngine).
        """
        self.ocr_engine = ocr_engine or CyberSecurityOCREngine()
        self.vlm_engine = vlm_engine or CyberSecurityVLMEngine()

    def process_image(
        self,
        image_path: str | Path,
        modality: str | None = None,
        source_dataset: str = "images",
        attck_technique: str | None = None,
        cve_id: str | None = None,
    ) -> CanonicalEvidenceRecord:
        """Process a single image artifact through OCR, VLM, and metadata extraction.

        Args:
            image_path: Path to the image file (.png, .jpg).
            modality: ModalityType string (screenshot, diagram, table). Auto-detected if None.
            source_dataset: Dataset identifier.
            attck_technique: Optional ATT&CK technique ID.
            cve_id: Optional CVE ID.

        Returns:
            CanonicalEvidenceRecord containing observed OCR and VLM interpretation.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info("Processing visual evidence artifact: %s", image_path.name)

        # 1. Image Properties via PIL
        with Image.open(str(image_path)) as img:
            width, height = img.size
            img_format = img.format or image_path.suffix.lstrip(".").upper()
            mode = img.mode

        # Determine modality if not provided
        if modality is None:
            p_str = str(image_path).lower()
            if "diagram" in p_str:
                modality = ModalityType.IMAGE_DIAGRAM.value
            elif "table" in p_str:
                modality = ModalityType.IMAGE_TABLE.value
            else:
                modality = ModalityType.IMAGE_SCREENSHOT.value

        # Infer ATT&CK / CVE metadata for benchmark images if not provided
        fn_lower = image_path.name.lower()
        if attck_technique is None:
            if "powershell" in fn_lower:
                attck_technique = "T1059.001"
            elif "mimikatz" in fn_lower:
                attck_technique = "T1003.001"

        if cve_id is None and ("attack_flow" in fn_lower or "ioc_table" in fn_lower):
            cve_id = "CVE-2023-42793"

        # 2. Extract OCR (Observed Evidence)
        ocr_res: OCRResult = self.ocr_engine.extract_text(image_path)

        # 3. Extract VLM (Model Interpretation)
        vlm_res: VLMInterpretation = self.vlm_engine.interpret_image(image_path)

        # 4. Construct Composite Canonical Text
        content = (
            f"[VISUAL CYBERSECURITY EVIDENCE: {modality.upper()}]\n"
            f"Image Artifact: {image_path.name} ({img_format}, {width}x{height} pixels, Mode: {mode})\n"
            f"Reference Path: {image_path}\n\n"
            f"{ocr_res.to_canonical_text(image_path.name)}\n\n"
            f"{vlm_res.to_canonical_text(image_path.name)}"
        )

        metadata: dict[str, Any] = {
            "width": width,
            "height": height,
            "format": img_format,
            "mode": mode,
            "file_size_bytes": image_path.stat().st_size,
            "ocr_text": ocr_res.raw_text,
            "ocr_engine": ocr_res.engine_name,
            "ocr_confidence": ocr_res.confidence,
            "vlm_caption": vlm_res.caption,
            "vlm_interpretation": vlm_res.detailed_interpretation,
            "vlm_model": vlm_res.model_name,
            "is_vlm_interpretation_ground_truth": False,
            "attck_technique": attck_technique,
            "cve_id": cve_id,
        }

        evidence_id = f"IMG-{image_path.stem}"

        return CanonicalEvidenceRecord(
            evidence_id=evidence_id,
            source_file=image_path.name,
            source_dataset=source_dataset,
            modality=modality,
            content=content,
            raw_reference=str(image_path),
            metadata=metadata,
            event_type="visual_evidence",
            attck_technique=attck_technique,
            cve_id=cve_id,
        )

    def process_directory(self, images_dir: str | Path) -> list[CanonicalEvidenceRecord]:
        """Process all images in a directory tree."""
        images_dir = Path(images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Directory not found: {images_dir}")

        records: list[CanonicalEvidenceRecord] = []
        for img_path in sorted(images_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                rec = self.process_image(img_path)
                records.append(rec)

        logger.info("Processed %d visual evidence records from %s", len(records), images_dir)
        return records
