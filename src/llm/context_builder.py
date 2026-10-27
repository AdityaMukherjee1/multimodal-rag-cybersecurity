"""Evidence Context Builder with strict Epistemic Segregation and Full Provenance.

Constructs structured LLM prompt context from top-K retrieved FusedRetrievalResult
objects while rigorously differentiating Observed Evidence (ground-truth logs,
exact OCR extractions, CVE advisories) from Model-Generated Interpretation
(BLIP VLM captions and synthetic inferences).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.multimodal.multimodal_retriever import FusedRetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class FormattedEvidenceContext:
    """Encapsulates the rendered context string and identified multimodal assets."""

    context_text: str
    evidence_ids: list[str]
    image_paths: list[str] = field(default_factory=list)
    total_items: int = 0
    observed_count: int = 0
    interpretation_count: int = 0


class EvidenceContextBuilder:
    """Constructs structured, epistemically segregated evidence contexts for LLM prompts."""

    def __init__(self, max_content_chars_per_item: int = 2500) -> None:
        self.max_content_chars = max_content_chars_per_item

    def build_context(
        self,
        query: str,
        results: list[FusedRetrievalResult],
    ) -> FormattedEvidenceContext:
        """Build a structured evidence block preserving all provenance and epistemic boundaries.

        Args:
            query: The user's investigation query.
            results: List of top-K fused retrieval candidates from MultimodalRetriever.

        Returns:
            FormattedEvidenceContext containing rendered text and multimodal metadata.
        """
        if not results:
            return FormattedEvidenceContext(
                context_text="No evidence retrieved from the cybersecurity knowledge base.",
                evidence_ids=[],
                image_paths=[],
                total_items=0,
            )

        blocks: list[str] = []
        evidence_ids: list[str] = []
        image_paths: list[str] = []
        observed_count = 0
        reference_count = 0
        interpretation_count = 0

        blocks.append("=" * 70)
        blocks.append(f"RETRIEVED FORENSIC EVIDENCE FOR QUERY: '{query}'")
        blocks.append(f"Total Candidates: {len(results)} | Evaluated Vector Stores: Text (BGE) + Visual (CLIP)")
        blocks.append("=" * 70)
        blocks.append("")

        for idx, item in enumerate(results, start=1):
            evidence_ids.append(item.evidence_id)
            is_visual = "image" in item.modality.lower()
            
            if item.evidence_nature == "OBSERVED_EVIDENCE":
                observed_count += 1
            elif item.evidence_nature == "REFERENCE_KNOWLEDGE":
                reference_count += 1
            else:
                interpretation_count += 1

            meta = item.metadata or {}
            timestamp = item.metadata.get("timestamp") or meta.get("created_at") or "N/A"
            attck = item.metadata.get("attck_technique") or meta.get("technique_id") or "N/A"
            cve = item.metadata.get("cve_id") or "N/A"
            event_type = item.metadata.get("event_type") or meta.get("category") or "N/A"

            prov_type = meta.get("provenance_type")
            if not prov_type:
                is_user_up = (
                    str(item.source_dataset).lower() == "user_upload"
                    or meta.get("is_user_provided", False)
                    or item.evidence_id.startswith("UPLOAD-")
                )
                prov_type = "USER_UPLOAD" if is_user_up else "KNOWLEDGE_BASE"

            lines: list[str] = [
                f"[{item.evidence_id}]",
                f"Provenance Source: {prov_type}",
                f"Evidence Nature: {item.evidence_nature}",
                f"Rank: #{item.rank or idx} | Fused Score: {item.fused_score:.4f} | Retrieval Modality: {item.retrieval_modality}",
                f"Source File: {item.source_file}",
                f"Source Dataset: {item.source_dataset}",
                f"Evidence Modality: {item.modality}",
                f"Raw Reference Path: {item.raw_reference}",
                f"Timestamp: {timestamp}",
                f"MITRE ATT&CK Mapping: {attck}",
                f"CVE Identifier: {cve}",
                f"Event Classification: {event_type}",
            ]

            # Collect image path if physical image exists
            if is_visual:
                raw_p = meta.get("disk_resolved_path", item.raw_reference)
                if Path(raw_p).exists() and raw_p not in image_paths:
                    image_paths.append(str(raw_p))

            # Epistemic Content Rendering
            lines.append("--- Evidence Content ---")

            if is_visual:
                # Segregate OCR text from BLIP interpretation
                lines.append(
                    self._format_visual_content(
                        content=item.content,
                        source_file=item.source_file,
                        meta=meta,
                    )
                )
            else:
                # Text/Log/Knowledge evidence
                if item.evidence_nature == "REFERENCE_KNOWLEDGE":
                    lines.append("[REFERENCE KNOWLEDGE: AUTHORITATIVE CYBERSECURITY RECORD]")
                elif item.evidence_nature == "MODEL_INTERPRETATION":
                    lines.append("[MODEL INTERPRETATION: SYSTEM GENERATED ANALYSIS]")
                else:
                    lines.append("[OBSERVED EVIDENCE: CANONICAL FORENSIC RECORD]")
                content_preview = item.content.strip()
                if len(content_preview) > self.max_content_chars:
                    content_preview = content_preview[: self.max_content_chars] + "\n... [TRUNCATED FOR CONTEXT LENGTH]"
                lines.append(content_preview)

            lines.append("")
            blocks.append("\n".join(lines))

        blocks.append("=" * 70)
        blocks.append("END OF RETRIEVED FORENSIC EVIDENCE")
        blocks.append("=" * 70)

        context_str = "\n".join(blocks)
        logger.info(
            "EvidenceContextBuilder constructed context: %d items (%d observed, %d interpretations), %d chars",
            len(results),
            observed_count,
            interpretation_count,
            len(context_str),
        )

        return FormattedEvidenceContext(
            context_text=context_str,
            evidence_ids=evidence_ids,
            image_paths=image_paths,
            total_items=len(results),
            observed_count=observed_count,
            interpretation_count=interpretation_count,
        )

    def _format_visual_content(
        self,
        content: str,
        source_file: str,
        meta: dict[str, Any],
    ) -> str:
        """Format visual evidence with explicit epistemic segregation between OCR and VLM."""
        # If content already has canonical tags, preserve and format them cleanly
        if "[OBSERVED EVIDENCE:" in content and "[MODEL-GENERATED INTERPRETATION" in content:
            return content.strip()

        # Otherwise format based on metadata fields
        parts: list[str] = []

        # 1. Observed OCR
        ocr_text = meta.get("ocr_text")
        if ocr_text:
            parts.append(f"[OBSERVED EVIDENCE: OCR TEXT EXTRACTED FROM {source_file}]")
            parts.append(ocr_text.strip())
        elif "[OBSERVED EVIDENCE:" in content:
            parts.append(content.split("[MODEL-GENERATED INTERPRETATION")[0].strip())
        else:
            parts.append(f"[OBSERVED EVIDENCE: OCR TEXT EXTRACTED FROM {source_file}]")
            parts.append("No explicit OCR text extracted or image contains purely visual diagram elements.")

        # 2. Model-Generated Interpretation
        vlm_caption = meta.get("vlm_caption") or meta.get("vlm_interpretation")
        parts.append("")
        parts.append(
            f"[MODEL-GENERATED INTERPRETATION - NOT GROUND TRUTH: VLM CAPTION FOR {source_file}]"
        )
        if vlm_caption:
            parts.append(f"Engine: Salesforce/blip-image-captioning-base\nCaption: {vlm_caption.strip()}")
        elif "[MODEL-GENERATED INTERPRETATION" in content:
            parts.append(content.split("[MODEL-GENERATED INTERPRETATION")[1].strip())
        else:
            parts.append(f"Visual content summary: {content.strip()}")

        return "\n".join(parts)
