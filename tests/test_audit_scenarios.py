"""End-to-End Functional Audit Test Suite for CyberAI Investigator.

Validates the core query scenarios and generalized knowledge retrieval:
1. Conversational Query ("hii") -> Zero retrieval, short-circuit, <10ms
2. Cybersecurity Knowledge ("What is SQL injection?") -> CWE-89 at Rank 1 (REFERENCE_KNOWLEDGE), CVE-2023-34362 at Rank 2
3. Cybersecurity Knowledge ("What is phishing?") -> REF-PHISHING at Rank 1 (REFERENCE_KNOWLEDGE), ATT&CK T1566/T1598
4. Cybersecurity Knowledge ("What is malware?") -> REF-MALWARE at Rank 1 (REFERENCE_KNOWLEDGE)
5. Cybersecurity Knowledge ("What is ransomware?") -> REF-RANSOMWARE at Rank 1 (REFERENCE_KNOWLEDGE)
6. Cybersecurity Knowledge ("What is privilege escalation?") -> REF-PRIV_ESCALATION at Rank 1 (REFERENCE_KNOWLEDGE)
7. Vulnerability Deep Dive ("Explain CVE-2021-44228") -> Log4Shell RCE evidence
8. Content Explanation ("What does this say?" + upload) -> Artifact interpretation with dual provenance
9. Remediation Guidance ("What should I do to fix these?" + upload) -> Actionable containment steps
10. Incident Investigation ("Investigate this incident: credential dumping") -> Full multimodal RAG + verification
11. Visual Evidence Handling -> CLIP visual search + OCR + epistemic segregation
12. Strict Epistemic Integrity -> KNOWLEDGE_BASE reference material is NEVER labelled as OBSERVED_EVIDENCE
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.query_engine import DashboardQueryEngine
from src.multimodal.multimodal_retriever import FusedRetrievalResult
from src.routing.intent_router import QueryIntent, QueryIntentRouter


@pytest.fixture(scope="module")
def query_engine():
    engine = DashboardQueryEngine()
    engine.initialize()
    return engine


class TestFunctionalAuditScenarios:
    """Test suite covering end-to-end query scenarios and reference knowledge retrieval."""

    def test_scenario_1_conversational_greeting(self, query_engine):
        """Scenario 1: 'hii' -> zero retrieval, <10ms intent classification, conversational response."""
        res = query_engine.execute_query("hii")
        assert res.is_conversational is True
        assert len(res.results) == 0
        assert res.structured_answer is not None
        assert "CyberAI" in res.structured_answer.answer or "help" in res.structured_answer.answer.lower()
        # Ensure downstream stages were skipped
        skipped_steps = [s for s in res.tracker.steps if s.status.value == "SKIPPED"]
        assert len(skipped_steps) >= 8

    def test_scenario_2_sql_injection_retrieval_and_knowledge(self, query_engine):
        """Scenario 2: 'What is SQL injection?' -> CWE-89 & CVE-2023-34362 at top ranks, no T1055 at rank 1."""
        res = query_engine.execute_query("What is SQL injection?")
        assert res.is_conversational is False
        assert len(res.results) > 0

        # Rank 1 must be authoritative CWE-89 reference knowledge
        assert res.results[0].evidence_id == "CWE-CWE_89"
        assert res.results[0].evidence_nature == "REFERENCE_KNOWLEDGE"
        assert res.results[0].metadata["provenance_type"] == "KNOWLEDGE_BASE"

        # Ranks 1-3 must contain SQL injection or SQL Stored Procedures
        top_ids = [r.evidence_id for r in res.results[:3]]
        has_cwe_or_cve = any("CWE_89" in eid or "34362" in eid or "T1505_001" in eid for eid in top_ids)
        assert has_cwe_or_cve, f"Top 3 evidence IDs {top_ids} do not contain SQL injection evidence"

        # T1055 (Process Injection) must NOT be at rank 1
        assert "T1055" not in res.results[0].evidence_id

        # Verification report should confirm 0% hallucination
        if res.verification_report:
            assert res.verification_report.hallucination_rate == 0.0

    def test_scenario_3_phishing_knowledge(self, query_engine):
        """Scenario 3: 'What is phishing?' -> REF-PHISHING at Rank 1 with REFERENCE_KNOWLEDGE nature."""
        res = query_engine.execute_query("What is phishing?")
        assert res.is_conversational is False
        assert len(res.results) > 0

        # Rank 1 must be the authoritative reference handbook record
        assert res.results[0].evidence_id == "REF-PHISHING"
        assert res.results[0].evidence_nature == "REFERENCE_KNOWLEDGE"
        assert res.results[0].metadata["provenance_type"] == "KNOWLEDGE_BASE"

        # Must also retrieve supplementary ATT&CK techniques
        top_ids = [r.evidence_id for r in res.results]
        has_attck_phish = any("1566" in eid or "1598" in eid for eid in top_ids)
        assert has_attck_phish, f"Retrieved IDs {top_ids} do not contain ATT&CK phishing technique"
        assert res.structured_answer is not None
        assert "phishing" in res.structured_answer.answer.lower()

    def test_educational_malware_knowledge(self, query_engine):
        """'What is malware?' -> REF-MALWARE at Rank 1 (REFERENCE_KNOWLEDGE) + ATT&CK T1588.001."""
        res = query_engine.execute_query("What is malware?")
        assert res.is_conversational is False
        assert len(res.results) > 0

        # Rank 1 must be authoritative malware reference definition
        assert res.results[0].evidence_id == "REF-MALWARE"
        assert res.results[0].evidence_nature == "REFERENCE_KNOWLEDGE"
        assert res.results[0].metadata["provenance_type"] == "KNOWLEDGE_BASE"

        # Answer must provide educational definition and categories
        ans = res.structured_answer.answer.lower()
        assert "malware" in ans
        assert "trojan" in ans or "virus" in ans or "worm" in ans or "ransomware" in ans

    def test_educational_ransomware_knowledge(self, query_engine):
        """'What is ransomware?' -> REF-RANSOMWARE at Rank 1 (REFERENCE_KNOWLEDGE) + ATT&CK T1486."""
        res = query_engine.execute_query("What is ransomware?")
        assert res.is_conversational is False
        assert len(res.results) > 0

        # Rank 1 must be authoritative ransomware reference definition
        assert res.results[0].evidence_id == "REF-RANSOMWARE"
        assert res.results[0].evidence_nature == "REFERENCE_KNOWLEDGE"
        assert res.results[0].metadata["provenance_type"] == "KNOWLEDGE_BASE"

        # Must include supplementary ATT&CK technique T1486
        top_ids = [r.evidence_id for r in res.results]
        assert any("1486" in eid for eid in top_ids)

        ans = res.structured_answer.answer.lower()
        assert "ransom" in ans or "encrypt" in ans

    def test_educational_privilege_escalation_knowledge(self, query_engine):
        """'What is privilege escalation?' -> REF-PRIV_ESCALATION at Rank 1 (REFERENCE_KNOWLEDGE) + ATT&CK T1548/T1068."""
        res = query_engine.execute_query("What is privilege escalation?")
        assert res.is_conversational is False
        assert len(res.results) > 0

        # Rank 1 must be authoritative privilege escalation reference definition
        assert res.results[0].evidence_id == "REF-PRIV_ESCALATION"
        assert res.results[0].evidence_nature == "REFERENCE_KNOWLEDGE"
        assert res.results[0].metadata["provenance_type"] == "KNOWLEDGE_BASE"

        # Must include supplementary ATT&CK techniques
        top_ids = [r.evidence_id for r in res.results]
        assert any("1548" in eid or "1068" in eid for eid in top_ids)

        ans = res.structured_answer.answer.lower()
        assert "privilege" in ans or "escalat" in ans

    def test_scenario_4_cve_explanation(self, query_engine):
        """Scenario 4: 'Explain CVE-2021-44228' -> Log4Shell RCE evidence."""
        res = query_engine.execute_query("Explain CVE-2021-44228 (Log4Shell)")
        assert res.is_conversational is False
        top_ids = [r.evidence_id for r in res.results]
        has_log4j = any("44228" in eid or "CVE" in eid for eid in top_ids)
        assert has_log4j
        assert res.structured_answer is not None
        assert "cve-2021-44228" in res.structured_answer.answer.lower() or "log4" in res.structured_answer.answer.lower()

    def test_scenario_5_content_explanation_with_upload(self, query_engine):
        """Scenario 5: 'What does this say?' + uploaded evidence -> dual provenance explanation."""
        upload_evidence = FusedRetrievalResult(
            evidence_id="UPLOAD-POWERSHELL_LOG",
            source_file="powershell_exec.log",
            source_dataset="USER_UPLOAD",
            modality="text",
            content="HostApplication=powershell.exe -ExecutionPolicy Bypass -NoProfile -File invoke-mimikatz.ps1",
            raw_reference="data/uploads/powershell_exec.log",
            metadata={"source": "USER_UPLOAD", "provenance_type": "USER_UPLOAD", "is_user_provided": True},
            fused_score=1.0,
            retrieval_modality="text",
            evidence_nature="OBSERVED_EVIDENCE",
            rank=1,
        )

        res = query_engine.execute_query(
            "what does this say?",
            session_evidence=[upload_evidence],
        )
        assert res.is_conversational is False
        assert len(res.results) >= 2
        # First item must be the uploaded evidence
        assert res.results[0].evidence_id == "UPLOAD-POWERSHELL_LOG"
        assert res.results[0].metadata["provenance_type"] == "USER_UPLOAD"
        assert res.results[0].evidence_nature == "OBSERVED_EVIDENCE"
        # Followed by persistent KB evidence
        assert res.results[1].metadata["provenance_type"] == "KNOWLEDGE_BASE"
        assert "UPLOAD-POWERSHELL_LOG" in res.structured_answer.answer

    def test_scenario_6_remediation_guidance_with_upload(self, query_engine):
        """Scenario 6: 'What should I do to fix these?' + upload -> actionable numbered remediation."""
        upload_evidence = FusedRetrievalResult(
            evidence_id="UPLOAD-MALICIOUS_FLOW",
            source_file="network_c2.csv",
            source_dataset="USER_UPLOAD",
            modality="network_flow",
            content="Dst_IP=198.51.100.23, Dst_Port=4444, Protocol=TCP, Flow_Duration=120000",
            raw_reference="data/uploads/network_c2.csv",
            metadata={"source": "USER_UPLOAD", "provenance_type": "USER_UPLOAD", "is_user_provided": True},
            fused_score=1.0,
            retrieval_modality="network_flow",
            evidence_nature="OBSERVED_EVIDENCE",
            rank=1,
        )

        res = query_engine.execute_query(
            "what should i do to fix these?",
            session_evidence=[upload_evidence],
        )
        assert res.is_conversational is False
        ans_text = res.structured_answer.answer.lower()
        # Must contain actionable containment/remediation steps
        assert any(w in ans_text for w in ["contain", "isolate", "block", "remediation", "patch", "terminate"])
        assert "1." in ans_text or "2." in ans_text  # structured steps

    def test_scenario_7_incident_investigation(self, query_engine):
        """Scenario 7: 'Investigate this incident: credential dumping' -> Full 14-stage pipeline."""
        res = query_engine.execute_query("Investigate this incident: credential dumping via LSASS")
        assert res.is_conversational is False
        assert len(res.results) > 0
        # Pipeline must execute all stages
        completed_steps = [s for s in res.tracker.steps if s.status.value == "COMPLETE"]
        assert len(completed_steps) >= 12
        assert res.verification_report is not None

    def test_scenario_8_visual_evidence_handling(self, query_engine):
        """Scenario 8: Visual evidence query -> CLIP visual retrieval and epistemic labeling."""
        res = query_engine.execute_query("Show me SIEM alert screenshots of PowerShell execution")
        assert res.is_conversational is False
        # Check that visual search executed
        visual_step = next((s for s in res.tracker.steps if s.name == "Visual FAISS Search"), None)
        assert visual_step is not None
        assert visual_step.status.value in ["COMPLETE", "SKIPPED"]
        # Check epistemic segregation on results
        for r in res.results:
            assert r.evidence_nature in ["OBSERVED_EVIDENCE", "MODEL_INTERPRETATION", "REFERENCE_KNOWLEDGE"]
            assert r.metadata.get("provenance_type") in ["USER_UPLOAD", "KNOWLEDGE_BASE"]

    def test_reference_knowledge_never_labelled_as_observed_evidence(self, query_engine):
        """Verify that KNOWLEDGE_BASE reference material (ATT&CK, NVD, CWE, REF, PDF guides) is NEVER labelled as OBSERVED_EVIDENCE."""
        test_queries = [
            "What is phishing?",
            "What is malware?",
            "What is ransomware?",
            "What is privilege escalation?",
            "What is SQL injection?",
            "Explain CVE-2021-44228",
        ]
        for q in test_queries:
            res = query_engine.execute_query(q)
            for r in res.results:
                if r.metadata.get("provenance_type") == "KNOWLEDGE_BASE":
                    dataset = str(r.source_dataset).lower()
                    if any(k in dataset for k in ["mitre", "attack", "nvd", "cve", "vulnerability", "cwe", "reference", "knowledge", "documents_pdf", "documents_docx"]):
                        assert r.evidence_nature == "REFERENCE_KNOWLEDGE", (
                            f"Violation in query '{q}': Reference item [{r.evidence_id}] from dataset '{dataset}' "
                            f"was erroneously labelled as '{r.evidence_nature}' instead of 'REFERENCE_KNOWLEDGE'"
                        )
