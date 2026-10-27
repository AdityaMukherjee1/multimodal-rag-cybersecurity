"""Comprehensive tests for query intent routing and natural assistant behavior.

Validates the full specification:
  1. Greetings ("hii", "hello", "thanks", "ok") → CONVERSATIONAL, zero retrieval, sub-10ms.
  2. General/Explanatory Questions ("What is phishing?", "explain privilege escalation",
     "what does this PowerShell command do?") → CYBERSECURITY_KNOWLEDGE path.
  3. Remediation Questions ("what should i do to fix these?", "how to fix this") → REMEDIATION path.
  4. Content Explanation ("what this says", "what does this say") → CYBERSECURITY_KNOWLEDGE path.
  5. Incident Investigations ("Investigate this PowerShell activity", "analyze this incident",
     "investigate these logs", "find the attack technique", "correlate these events",
     "why did this compromise happen?") → INVESTIGATION path.
  6. File/Image Attachments:
     - "hi" + uploaded evidence → binds upload, analyzes evidence, not plain chitchat bypass.
     - "what this says" + uploaded evidence → inspects and explains uploaded artifact.
     - "what should i do to fix these?" + uploaded evidence → remediation tailored to upload.
     - "investigate this PowerShell activity" + uploaded evidence → full multimodal investigation.
  7. End-to-end execution through DashboardQueryEngine:
     - Conversational inputs bypass expensive retrieval, models, and verification.
     - Evidence-based queries run authentic retrieval and produce grounded structured answers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.multimodal.multimodal_retriever import FusedRetrievalResult
from src.routing.intent_router import IntentClassification, QueryIntent, QueryIntentRouter


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────

@pytest.fixture
def router():
    return QueryIntentRouter()


@pytest.fixture
def mock_upload_evidence():
    return FusedRetrievalResult(
        evidence_id="UPLOAD-TEST_LOG",
        source_file="auth_security.evtx",
        source_dataset="USER_UPLOAD",
        modality="text",
        content="EventID 4625: An account failed to log on. Account Name: Administrator. Workstation: WORKSTATION-01.",
        raw_reference="data/uploads/auth_security.evtx",
        metadata={"provenance_type": "USER_UPLOAD", "is_user_provided": True},
        fused_score=1.0,
        text_score=1.0,
        visual_score=None,
        retrieval_modality="text",
        evidence_nature="OBSERVED_EVIDENCE",
        rank=1,
    )


# ────────────────────────────────────────────────────────────
# 1. Greetings & Conversational Queries (No Evidence)
# ────────────────────────────────────────────────────────────

class TestGreetingDetection:
    def test_greeting_hii(self, router):
        """'hii' must be classified as CONVERSATIONAL with zero retrieval needed."""
        result = router.classify("hii")
        assert result.intent == QueryIntent.CONVERSATIONAL
        assert result.confidence >= 0.90
        assert "greeting" in result.reason.lower() or "conversational" in result.reason.lower()

    def test_greeting_hello(self, router):
        """'hello' must be classified as CONVERSATIONAL."""
        result = router.classify("hello")
        assert result.intent == QueryIntent.CONVERSATIONAL
        assert result.confidence >= 0.90

    def test_greeting_hey(self, router):
        result = router.classify("hey")
        assert result.intent == QueryIntent.CONVERSATIONAL

    def test_greeting_howdy(self, router):
        result = router.classify("howdy")
        assert result.intent == QueryIntent.CONVERSATIONAL

    def test_greeting_good_morning(self, router):
        result = router.classify("good morning")
        assert result.intent == QueryIntent.CONVERSATIONAL

    def test_thanks(self, router):
        """'thanks' must be classified as CONVERSATIONAL."""
        result = router.classify("thanks")
        assert result.intent == QueryIntent.CONVERSATIONAL
        assert result.confidence >= 0.90

    def test_thank_you(self, router):
        result = router.classify("thank you")
        assert result.intent == QueryIntent.CONVERSATIONAL

    def test_ok(self, router):
        result = router.classify("ok")
        assert result.intent == QueryIntent.CONVERSATIONAL

    def test_sure(self, router):
        result = router.classify("sure")
        assert result.intent == QueryIntent.CONVERSATIONAL

    def test_bye(self, router):
        result = router.classify("bye")
        assert result.intent == QueryIntent.CONVERSATIONAL

    def test_who_are_you(self, router):
        result = router.classify("who are you")
        assert result.intent == QueryIntent.CONVERSATIONAL


# ────────────────────────────────────────────────────────────
# 2. General / Explanatory Cybersecurity Questions
# ────────────────────────────────────────────────────────────

class TestCyberKnowledgeDetection:
    def test_what_is_phishing(self, router):
        """'What is phishing?' must trigger cybersecurity knowledge path, not conversational."""
        result = router.classify("What is phishing?")
        assert result.intent == QueryIntent.CYBERSECURITY_KNOWLEDGE
        assert result.confidence >= 0.80

    def test_what_does_this_powershell_command_do(self, router):
        """'what does this PowerShell command do?' must route to explanatory knowledge."""
        result = router.classify("what does this PowerShell command do?")
        assert result.intent == QueryIntent.CYBERSECURITY_KNOWLEDGE

    def test_explain_privilege_escalation(self, router):
        """'explain privilege escalation' must route to cybersecurity knowledge."""
        result = router.classify("explain privilege escalation")
        assert result.intent == QueryIntent.CYBERSECURITY_KNOWLEDGE

    def test_explain_cve(self, router):
        result = router.classify("What is CVE-2021-44228?")
        assert result.intent == QueryIntent.CYBERSECURITY_KNOWLEDGE

    def test_ransomware_explanation(self, router):
        result = router.classify("How does ransomware work?")
        assert result.intent == QueryIntent.CYBERSECURITY_KNOWLEDGE

    def test_short_cyber_terms(self, router):
        """1-2 word cybersecurity terms must route to knowledge, not conversational."""
        for term in ["WannaCry", "DDoS", "Metasploit", "Stuxnet", "Kerberoasting", "CWE-79"]:
            res = router.classify(term)
            assert res.intent in (QueryIntent.CYBERSECURITY_KNOWLEDGE, QueryIntent.INVESTIGATION), f"Failed for {term}: got {res.intent}"


# ────────────────────────────────────────────────────────────
# 3. Remediation Questions
# ────────────────────────────────────────────────────────────

class TestRemediationDetection:
    def test_what_should_i_do_to_fix_these(self, router):
        """'what should i do to fix these?' must route to REMEDIATION."""
        result = router.classify("what should i do to fix these?")
        assert result.intent == QueryIntent.REMEDIATION
        assert result.confidence >= 0.85

    def test_how_do_i_fix_this(self, router):
        result = router.classify("how do i fix this?")
        assert result.intent == QueryIntent.REMEDIATION

    def test_remediation_steps(self, router):
        result = router.classify("what are the recommended remediation steps?")
        assert result.intent == QueryIntent.REMEDIATION

    def test_how_to_mitigate(self, router):
        result = router.classify("how to mitigate this vulnerability?")
        assert result.intent == QueryIntent.REMEDIATION


# ────────────────────────────────────────────────────────────
# 4. Content Explanation Questions
# ────────────────────────────────────────────────────────────

class TestContentExplanationDetection:
    def test_what_this_says(self, router):
        """'what this says' must be classified as explanatory, not conversational."""
        result = router.classify("what this says")
        assert result.intent == QueryIntent.CYBERSECURITY_KNOWLEDGE

    def test_what_does_this_say(self, router):
        result = router.classify("what does this say")
        assert result.intent == QueryIntent.CYBERSECURITY_KNOWLEDGE

    def test_explain_this_screenshot(self, router):
        result = router.classify("explain this screenshot")
        assert result.intent == QueryIntent.CYBERSECURITY_KNOWLEDGE


# ────────────────────────────────────────────────────────────
# 5. Incident Investigation Queries
# ────────────────────────────────────────────────────────────

class TestInvestigationDetection:
    def test_investigate_powershell_activity(self, router):
        """'Investigate this PowerShell activity' must trigger INVESTIGATION."""
        result = router.classify("Investigate this PowerShell activity")
        assert result.intent == QueryIntent.INVESTIGATION
        assert result.confidence >= 0.80

    def test_analyze_this_incident(self, router):
        """'analyze this incident' must trigger INVESTIGATION."""
        result = router.classify("analyze this incident")
        assert result.intent == QueryIntent.INVESTIGATION

    def test_investigate_these_logs(self, router):
        """'investigate these logs' must trigger INVESTIGATION."""
        result = router.classify("investigate these logs")
        assert result.intent == QueryIntent.INVESTIGATION

    def test_find_the_attack_technique(self, router):
        """'find the attack technique' must trigger INVESTIGATION."""
        result = router.classify("find the attack technique")
        assert result.intent == QueryIntent.INVESTIGATION

    def test_correlate_these_events(self, router):
        """'correlate these events' must trigger INVESTIGATION."""
        result = router.classify("correlate these events")
        assert result.intent == QueryIntent.INVESTIGATION

    def test_why_did_this_compromise_happen(self, router):
        """'why did this compromise happen?' must trigger INVESTIGATION."""
        result = router.classify("why did this compromise happen?")
        assert result.intent == QueryIntent.INVESTIGATION


# ────────────────────────────────────────────────────────────
# 6. File / Image Attachment Integration
# ────────────────────────────────────────────────────────────

class TestFileUploadIntegration:
    def test_hi_with_uploaded_evidence(self, router):
        """'hi' with uploaded evidence must NOT be bypassed as pure chitchat."""
        result = router.classify("hi", has_file_attachment=True)
        assert result.intent == QueryIntent.INVESTIGATION
        assert "upload" in result.reason.lower() or "attached" in result.reason.lower()

    def test_what_this_says_with_uploaded_evidence(self, router):
        """'what this says' with uploaded evidence must route to content explanation."""
        result = router.classify("what this says", has_file_attachment=True)
        assert result.intent in (QueryIntent.CYBERSECURITY_KNOWLEDGE, QueryIntent.INVESTIGATION)

    def test_what_should_i_do_to_fix_these_with_evidence(self, router):
        """'what should i do to fix these?' with evidence must route to REMEDIATION."""
        result = router.classify("what should i do to fix these?", has_file_attachment=True)
        assert result.intent == QueryIntent.REMEDIATION

    def test_investigate_with_uploaded_evidence(self, router):
        """'Investigate this PowerShell activity' + file must route to INVESTIGATION."""
        result = router.classify("Investigate this PowerShell activity", has_file_attachment=True)
        assert result.intent == QueryIntent.INVESTIGATION

    def test_session_has_evidence_persists_context(self, router):
        """Subsequent queries in a session with evidence must recognize active evidence."""
        result = router.classify("what should i do to fix these?", session_has_evidence=True)
        assert result.intent == QueryIntent.REMEDIATION


# ────────────────────────────────────────────────────────────
# 7. Classification Latency
# ────────────────────────────────────────────────────────────

class TestClassificationPerformance:
    def test_classification_under_10ms(self, router):
        """Classification must be sub-10ms (rule-based, no GPU, zero overhead)."""
        queries = [
            ("hii", False),
            ("hello", False),
            ("thanks", False),
            ("What is phishing?", False),
            ("what should i do to fix these?", False),
            ("Investigate this PowerShell activity", False),
            ("what this says", True),
            ("hi", True),
        ]
        start = time.perf_counter()
        for _ in range(50):
            for q, has_file in queries:
                router.classify(q, has_file_attachment=has_file)
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / (50 * len(queries))
        assert avg_ms < 10.0, f"Average latency {avg_ms:.2f}ms exceeds 10ms"


# ────────────────────────────────────────────────────────────
# 8. End-to-End Execution via DashboardQueryEngine
# ────────────────────────────────────────────────────────────

class TestEndToEndDashboardEngine:
    def test_hii_shortcircuit_zero_retrieval(self):
        """'hii' via DashboardQueryEngine must return 0 evidence, 0 claims, no verification, fast latency."""
        from frontend.query_engine import DashboardQueryEngine, QueryExecutionResult

        engine = DashboardQueryEngine()
        result = engine.execute_query(query="hii")

        assert isinstance(result, QueryExecutionResult)
        assert result.is_conversational is True
        assert len(result.results) == 0, "Conversational query must have zero retrieved evidence"
        assert result.verification_report is None, "Conversational query must have no verification"
        assert result.structured_answer is not None
        assert len(result.structured_answer.claims) == 0
        assert "cyberai" in result.structured_answer.answer.lower() or "investigator" in result.structured_answer.answer.lower()
        assert result.tracker.metrics.total_latency_ms < 5000, "Conversational query must complete in <5s"
        assert result.error is None

    def test_what_is_phishing_runs_rag_path(self):
        """'What is phishing?' must execute RAG path and produce structured answer."""
        from frontend.query_engine import DashboardQueryEngine

        engine = DashboardQueryEngine()
        result = engine.execute_query(query="What is phishing?")

        assert result.is_conversational is False
        assert result.structured_answer is not None
        assert "phishing" in result.structured_answer.answer.lower()

    def test_remediation_query_produces_actionable_steps(self, mock_upload_evidence):
        """'what should i do to fix these?' with evidence produces structured remediation."""
        from frontend.query_engine import DashboardQueryEngine

        engine = DashboardQueryEngine()
        result = engine.execute_query(
            query="what should i do to fix these?",
            session_evidence=[mock_upload_evidence],
        )

        assert result.is_conversational is False
        assert result.structured_answer is not None
        ans_text = result.structured_answer.answer.lower()
        assert any(k in ans_text for k in ["remediation", "containment", "isolate", "credential", "patch"])
        assert len(result.results) > 0

    def test_hi_with_upload_analyzes_evidence(self, mock_upload_evidence):
        """'hi' with uploaded evidence must analyze the evidence and not be bypassed."""
        from frontend.query_engine import DashboardQueryEngine

        engine = DashboardQueryEngine()
        result = engine.execute_query(
            query="hi",
            session_evidence=[mock_upload_evidence],
        )

        assert result.is_conversational is False
        assert result.structured_answer is not None
        ans_text = result.structured_answer.answer.lower()
        assert "upload" in ans_text or "evidence" in ans_text or "telemetry" in ans_text
        assert len(result.results) > 0
