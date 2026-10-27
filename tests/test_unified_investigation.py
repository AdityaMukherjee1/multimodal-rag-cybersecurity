"""Unit and integration tests for Unified Investigation with User Uploads & Knowledge Base.

Validates that:
1. User-uploaded evidence is ingested with USER_UPLOAD provenance.
2. The EvidenceContextBuilder includes both USER_UPLOAD and KNOWLEDGE_BASE candidates in the same context.
3. Provenance labels (USER_UPLOAD vs KNOWLEDGE_BASE) are strictly preserved.
4. The LLM generates cross-correlated claims citing both sources.
5. HybridEvidenceVerifier verifies claims across both provenance origins.
"""

from pathlib import Path
import pytest

from src.llm.context_builder import EvidenceContextBuilder
from src.llm.mock_provider import MockLLMProvider
from src.llm.prompt_templates import INVESTIGATION_SYSTEM_PROMPT, build_investigation_user_prompt
from src.multimodal.multimodal_retriever import FusedRetrievalResult
from src.verification.evidence_verifier import HybridEvidenceVerifier
from src.verification.schema import VerificationStatus
from frontend.query_engine import DashboardQueryEngine


class TestUnifiedInvestigation:
    """Test suite for integrated investigations across user uploads and persistent knowledge base."""

    @pytest.fixture
    def sample_user_upload_evidence(self) -> FusedRetrievalResult:
        return FusedRetrievalResult(
            evidence_id="UPLOAD-MIMIKATZ",
            source_file="incident_terminal.png",
            source_dataset="USER_UPLOAD",
            modality="image_screenshot",
            content="[USER-UPLOADED FORENSIC SCREENSHOT]\nOCR: mimikatz.exe sekurlsa::logonpasswords privilege::debug",
            raw_reference="data/uploads/incident_terminal.png",
            metadata={
                "source": "USER_UPLOAD",
                "provenance_type": "USER_UPLOAD",
                "file_type": ".png",
                "is_user_provided": True,
            },
            fused_score=1.0,
            text_score=1.0,
            visual_score=1.0,
            retrieval_modality="multimodal",
            evidence_nature="OBSERVED_EVIDENCE",
            rank=1,
        )

    @pytest.fixture
    def sample_kb_evidence(self) -> FusedRetrievalResult:
        return FusedRetrievalResult(
            evidence_id="ATTCK-T1003",
            source_file="enterprise-attack.json",
            source_dataset="mitre_attack",
            modality="knowledge",
            content="[MITRE ATT&CK T1003 OS Credential Dumping]\nAdversaries dump credentials to harvest security identifiers from LSASS memory.",
            raw_reference="data/raw/mitre_attack/enterprise-attack.json",
            metadata={
                "source": "KNOWLEDGE_BASE",
                "provenance_type": "KNOWLEDGE_BASE",
                "attck_technique": "T1003",
            },
            fused_score=0.88,
            text_score=0.88,
            visual_score=None,
            retrieval_modality="text",
            evidence_nature="OBSERVED_EVIDENCE",
            rank=2,
        )

    def test_context_builder_includes_both_provenance_types(
        self,
        sample_user_upload_evidence: FusedRetrievalResult,
        sample_kb_evidence: FusedRetrievalResult,
    ):
        builder = EvidenceContextBuilder()
        formatted = builder.build_context(
            query="Analyze uploaded screenshot and correlate with credential dumping",
            results=[sample_user_upload_evidence, sample_kb_evidence],
        )

        assert "Provenance Source: USER_UPLOAD" in formatted.context_text
        assert "Provenance Source: KNOWLEDGE_BASE" in formatted.context_text
        assert "UPLOAD-MIMIKATZ" in formatted.evidence_ids
        assert "ATTCK-T1003" in formatted.evidence_ids
        assert formatted.total_items == 2

    def test_system_prompt_enforces_unified_investigation(self):
        assert "UNIFIED INVESTIGATION & PROVENANCE ATTRIBUTION" in INVESTIGATION_SYSTEM_PROMPT
        assert "USER_UPLOAD" in INVESTIGATION_SYSTEM_PROMPT
        assert "KNOWLEDGE_BASE" in INVESTIGATION_SYSTEM_PROMPT

    def test_mock_llm_synthesizes_unified_claims(
        self,
        sample_user_upload_evidence: FusedRetrievalResult,
        sample_kb_evidence: FusedRetrievalResult,
    ):
        builder = EvidenceContextBuilder()
        formatted = builder.build_context(
            query="What activity is shown in the uploaded evidence?",
            results=[sample_user_upload_evidence, sample_kb_evidence],
        )

        mock_llm = MockLLMProvider()
        response = mock_llm.generate_response(
            prompt="What activity is shown in the uploaded evidence?",
            context=formatted.context_text,
        )

        assert response.structured is not None
        structured = response.structured
        assert len(structured.claims) >= 2

        # Check claim attribution: one citing upload, one citing both/KB
        cited_all = [cid for c in structured.claims for cid in c.supporting_evidence_ids]
        assert any("UPLOAD" in cid for cid in cited_all)
        assert any("ATTCK" in cid for cid in cited_all)

    def test_verifier_audits_claims_across_upload_and_kb(
        self,
        sample_user_upload_evidence: FusedRetrievalResult,
        sample_kb_evidence: FusedRetrievalResult,
    ):
        verifier = HybridEvidenceVerifier()
        builder = EvidenceContextBuilder()
        formatted = builder.build_context(
            query="Verify uploaded evidence with credential dumping technique",
            results=[sample_user_upload_evidence, sample_kb_evidence],
        )

        mock_llm = MockLLMProvider()
        response = mock_llm.generate_response(
            prompt="Verify uploaded evidence with credential dumping technique",
            context=formatted.context_text,
        )

        report = verifier.verify_all(
            claims=response.structured.claims,
            evidence=[sample_user_upload_evidence, sample_kb_evidence],
            query="Verify uploaded evidence with credential dumping technique",
            raw_answer=response.structured.answer,
        )

        assert report.total_claims >= 2
        assert (report.supported_count + report.partially_supported_count) >= 1
        assert report.overall_grounding_score > 0.0

    def test_query_engine_unified_execution(
        self,
        sample_user_upload_evidence: FusedRetrievalResult,
    ):
        engine = DashboardQueryEngine()
        engine.initialize(llm_provider_override="mock")

        res = engine.execute_query(
            query="What evidence indicates credential dumping?",
            top_k=4,
            session_evidence=[sample_user_upload_evidence],
        )

        # Both USER_UPLOAD and KNOWLEDGE_BASE candidates must be present in fused results
        sources = [r.source_dataset for r in res.results]
        assert "USER_UPLOAD" in sources
        assert any(s != "USER_UPLOAD" for s in sources)

        # Evidence context must include both
        assert "UPLOAD-MIMIKATZ" in res.evidence_context
        assert "Provenance Source: USER_UPLOAD" in res.evidence_context
        assert "Provenance Source: KNOWLEDGE_BASE" in res.evidence_context

        # Structured answer must have claims attributed to both
        assert res.structured_answer is not None
        cited = [cid for c in res.structured_answer.claims for cid in c.supporting_evidence_ids]
        assert any("UPLOAD" in cid for cid in cited)
        assert any("UPLOAD" not in cid for cid in cited)
