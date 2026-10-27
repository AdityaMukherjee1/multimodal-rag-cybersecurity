"""Unit tests for HybridEvidenceVerifier across all 5 verification classifications."""

from src.llm.schema import Claim
from src.multimodal.multimodal_retriever import FusedRetrievalResult
from src.verification.evidence_verifier import HybridEvidenceVerifier
from src.verification.schema import VerificationStatus


class TestHybridEvidenceVerifier:
    def setup_method(self):
        self.verifier = HybridEvidenceVerifier()

        self.ev_windows_ps = FusedRetrievalResult(
            evidence_id="DOCX-001",
            source_file="soc_briefing.docx",
            source_dataset="soc_reports",
            modality="docx",
            content="Host WORKSTATION-01 spawned powershell.exe with -ExecutionPolicy Bypass. PID 4528. MITRE ATT&CK T1059.001.",
            raw_reference="data/sample/soc_briefing.docx",
            metadata={"attck_technique": "T1059.001", "event_type": "ProcessCreate"},
            fused_score=0.88,
            rank=1,
            evidence_nature="OBSERVED_EVIDENCE",
        )

        self.ev_mimikatz_ocr = FusedRetrievalResult(
            evidence_id="IMG-001",
            source_file="terminal_mimikatz.png",
            source_dataset="screenshots",
            modality="image_screenshot",
            content="Alert terminal showing mimikatz execution",
            raw_reference="data/raw/images/screenshots/terminal_mimikatz.png",
            metadata={
                "ocr_text": "mimikatz # privilege::debug\nPrivilege '20' OK\nmimikatz # sekurlsa::logonpasswords\nMITRE T1003.001",
                "vlm_caption": "Command line interface window with text commands on black background.",
                "attck_technique": "T1003.001",
            },
            fused_score=0.79,
            rank=2,
            evidence_nature="OBSERVED_EVIDENCE",
        )

        self.ev_pure_blip = FusedRetrievalResult(
            evidence_id="IMG-002",
            source_file="diagram.png",
            source_dataset="diagrams",
            modality="image_diagram",
            content="Attack flow diagram",
            raw_reference="data/raw/images/diagram.png",
            metadata={
                "ocr_text": "",
                "vlm_caption": "Architecture diagram illustrating multi-stage adversary C2 beaconing to external IP.",
            },
            fused_score=0.65,
            rank=3,
            evidence_nature="MODEL_INTERPRETATION",
        )

    def test_supported_text_claim(self):
        claim = Claim(
            claim_id="CLAIM-001",
            text="PowerShell was spawned with -ExecutionPolicy Bypass (T1059.001) on WORKSTATION-01.",
            supporting_evidence_ids=["DOCX-001"],
        )
        res = self.verifier.verify_claim(claim, [self.ev_windows_ps])
        assert res.status == VerificationStatus.SUPPORTED
        assert res.confidence_score >= 0.70
        assert res.is_observed_fact is True
        assert len(res.matched_evidence_snippets) > 0

    def test_supported_multimodal_ocr_claim(self):
        claim = Claim(
            claim_id="CLAIM-002",
            text="Terminal capture shows Mimikatz execution of privilege::debug and sekurlsa::logonpasswords (T1003.001).",
            supporting_evidence_ids=["IMG-001"],
        )
        res = self.verifier.verify_claim(claim, [self.ev_mimikatz_ocr])
        assert res.status == VerificationStatus.SUPPORTED
        assert res.confidence_score >= 0.70
        assert res.is_observed_fact is True

    def test_partially_supported_extrapolation_claim(self):
        claim = Claim(
            claim_id="CLAIM-003",
            text="PowerShell was executed with -ExecutionPolicy Bypass (T1059.001) and exfiltrated sensitive files to an unknown Dropbox account.",
            supporting_evidence_ids=["DOCX-001"],
        )
        res = self.verifier.verify_claim(claim, [self.ev_windows_ps])
        assert res.status == VerificationStatus.PARTIALLY_SUPPORTED
        assert "extrapolated" in res.reasoning.lower() or "unconfirmed" in res.reasoning.lower()

    def test_unsupported_hallucination_claim(self):
        claim = Claim(
            claim_id="CLAIM-004",
            text="The attacker dumped master browser passwords from Google Chrome and decrypted encrypted credit card tokens.",
            supporting_evidence_ids=["DOCX-001"],
        )
        res = self.verifier.verify_claim(claim, [self.ev_windows_ps])
        assert res.status in (VerificationStatus.UNSUPPORTED, VerificationStatus.INSUFFICIENT_EVIDENCE)

    def test_contradicted_claim(self):
        claim = Claim(
            claim_id="CLAIM-005",
            text="The attack was executed entirely on an Ubuntu Linux server using Bash scripts.",
            supporting_evidence_ids=["DOCX-001"],
        )
        res = self.verifier.verify_claim(claim, [self.ev_windows_ps])
        assert res.status == VerificationStatus.CONTRADICTED
        assert "contradict" in res.reasoning.lower()

    def test_epistemic_warning_on_pure_blip_interpretation(self):
        claim = Claim(
            claim_id="CLAIM-006",
            text="Adversary C2 beaconing to external IP was observed in architecture diagram.",
            supporting_evidence_ids=["IMG-002"],
        )
        res = self.verifier.verify_claim(claim, [self.ev_pure_blip])
        # Must not be marked as fully SUPPORTED because it only relies on BLIP interpretation without OCR telemetry
        assert res.status == VerificationStatus.PARTIALLY_SUPPORTED
        assert res.is_observed_fact is False
        assert res.epistemic_warning is not None
        assert "BLIP" in res.epistemic_warning
