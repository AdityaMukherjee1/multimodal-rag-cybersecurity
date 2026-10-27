"""Unit tests for EvidenceContextBuilder.

Ensures strict preservation of canonical forensic provenance,
epistemic segregation between observed telemetry and model interpretation,
and multimodal context formatting.
"""

from src.llm.context_builder import EvidenceContextBuilder
from src.multimodal.multimodal_retriever import FusedRetrievalResult


class TestEvidenceContextBuilder:
    def test_build_empty_context(self):
        builder = EvidenceContextBuilder()
        ctx = builder.build_context(query="test query", results=[])
        assert ctx.total_items == 0
        assert "No evidence retrieved" in ctx.context_text
        assert len(ctx.evidence_ids) == 0

    def test_build_heterogeneous_context_provenance_and_epistemic_segregation(self):
        builder = EvidenceContextBuilder()

        text_evidence = FusedRetrievalResult(
            evidence_id="DOCX-soc_incident_investigation_briefing-003",
            source_file="soc_incident_investigation_briefing.docx",
            source_dataset="sample_docx",
            modality="docx",
            content="Evidence confirms execution of powershell.exe with -ExecutionPolicy Bypass.",
            raw_reference="data/sample/docx/soc_incident_investigation_briefing.docx",
            metadata={
                "attck_technique": "T1059.001",
                "cve_id": None,
                "timestamp": "2024-10-14T08:30:00Z",
                "event_type": "ProcessCreate",
            },
            fused_score=0.8523,
            rank=1,
            retrieval_modality="text",
            evidence_nature="OBSERVED_EVIDENCE",
        )

        visual_evidence = FusedRetrievalResult(
            evidence_id="IMG-siem_alert_powershell_execution",
            source_file="siem_alert_powershell_execution.png",
            source_dataset="sample_images",
            modality="image_screenshot",
            content="Alert Banner: Critical Severity. Process lineage explorer.exe -> powershell.exe",
            raw_reference="data/raw/images/screenshots/siem_alert_powershell_execution.png",
            metadata={
                "attck_technique": "T1059.001",
                "cve_id": None,
                "timestamp": "2024-10-14T08:31:00Z",
                "ocr_text": "CRITICAL ALERT: Suspicious PowerShell Process Creation - ExecutionPolicy Bypass\nPID: 4528",
                "vlm_caption": "SIEM dashboard interface displaying critical alert window with process hierarchy.",
            },
            fused_score=0.6210,
            rank=2,
            retrieval_modality="visual",
            evidence_nature="MODEL_INTERPRETATION",
        )

        ctx = builder.build_context(
            query="What evidence indicates PowerShell bypass?",
            results=[text_evidence, visual_evidence],
        )

        # Basic counts
        assert ctx.total_items == 2
        assert "DOCX-soc_incident_investigation_briefing-003" in ctx.evidence_ids
        assert "IMG-siem_alert_powershell_execution" in ctx.evidence_ids

        rendered = ctx.context_text

        # Provenance verification
        assert "DOCX-soc_incident_investigation_briefing-003" in rendered
        assert "soc_incident_investigation_briefing.docx" in rendered
        assert "sample_docx" in rendered
        assert "data/sample/docx/soc_incident_investigation_briefing.docx" in rendered
        assert "T1059.001" in rendered
        assert "2024-10-14T08:30:00Z" in rendered

        # Epistemic segregation verification
        # 1. Text evidence is observed evidence
        assert "[OBSERVED EVIDENCE: CANONICAL FORENSIC RECORD]" in rendered
        assert "powershell.exe with -ExecutionPolicy Bypass" in rendered

        # 2. Visual evidence segregates OCR from BLIP interpretation
        assert "[OBSERVED EVIDENCE: OCR TEXT EXTRACTED FROM siem_alert_powershell_execution.png]" in rendered
        assert "CRITICAL ALERT: Suspicious PowerShell Process Creation" in rendered
        assert "[MODEL-GENERATED INTERPRETATION - NOT GROUND TRUTH: VLM CAPTION FOR siem_alert_powershell_execution.png]" in rendered
        assert "Salesforce/blip-image-captioning-base" in rendered
        assert "SIEM dashboard interface displaying critical alert window" in rendered
