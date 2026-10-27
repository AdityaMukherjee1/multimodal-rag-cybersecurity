"""Unit tests for Phase 4 Verification Schema and Report Metrics."""

from src.verification.schema import ClaimVerificationResult, VerificationReport, VerificationStatus


class TestVerificationSchema:
    def test_verification_status_values(self):
        assert VerificationStatus.SUPPORTED.value == "SUPPORTED"
        assert VerificationStatus.PARTIALLY_SUPPORTED.value == "PARTIALLY_SUPPORTED"
        assert VerificationStatus.UNSUPPORTED.value == "UNSUPPORTED"
        assert VerificationStatus.CONTRADICTED.value == "CONTRADICTED"
        assert VerificationStatus.INSUFFICIENT_EVIDENCE.value == "INSUFFICIENT_EVIDENCE"

    def test_claim_verification_result_serialization(self):
        res = ClaimVerificationResult(
            claim_id="CLAIM-001",
            claim_text="PowerShell execution detected.",
            supporting_evidence_ids=["DOCX-001"],
            status=VerificationStatus.SUPPORTED,
            confidence_score=0.92,
            reasoning="Direct match with DOCX briefing.",
            matched_evidence_snippets=["powershell.exe -ExecutionPolicy Bypass"],
            modality="docx",
            is_observed_fact=True,
            provenance_sources=["data/sample/briefing.docx"],
        )

        d = res.to_dict()
        assert d["status"] == "SUPPORTED"
        assert d["claim_id"] == "CLAIM-001"
        assert d["confidence_score"] == 0.92

        restored = ClaimVerificationResult.from_dict(d)
        assert restored.status == VerificationStatus.SUPPORTED
        assert restored.claim_id == res.claim_id
        assert restored.is_observed_fact is True

    def test_verification_report_metrics(self):
        r1 = ClaimVerificationResult(
            claim_id="C1", claim_text="Fact 1", status=VerificationStatus.SUPPORTED
        )
        r2 = ClaimVerificationResult(
            claim_id="C2", claim_text="Fact 2", status=VerificationStatus.PARTIALLY_SUPPORTED
        )
        r3 = ClaimVerificationResult(
            claim_id="C3", claim_text="Fact 3", status=VerificationStatus.UNSUPPORTED
        )

        report = VerificationReport(
            query="test query",
            claim_results=[r1, r2, r3],
            total_claims=3,
            supported_count=1,
            partially_supported_count=1,
            unsupported_count=1,
            contradicted_count=0,
            insufficient_evidence_count=0,
            hallucination_rate=round(1 / 3, 4),
            overall_grounding_score=round((1.0 + 0.5 + 0.0) / 3, 4),
        )

        assert report.total_claims == 3
        assert report.supported_count == 1
        assert report.hallucination_rate > 0.3
        assert report.overall_grounding_score == 0.5
