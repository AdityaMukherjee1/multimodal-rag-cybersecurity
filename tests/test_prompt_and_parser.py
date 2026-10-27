"""Unit tests for prompt formatting and structured output parsing.

Tests schema extraction, claim-level attribution, markdown unwrapping,
and resilient heuristic fallback.
"""

from src.llm.output_parser import InvestigationOutputParser
from src.llm.prompt_templates import INVESTIGATION_SYSTEM_PROMPT, build_investigation_user_prompt


class TestPromptTemplates:
    def test_system_prompt_enforces_principles(self):
        assert "GROUNDING" in INVESTIGATION_SYSTEM_PROMPT
        assert "EPISTEMIC SEGREGATION" in INVESTIGATION_SYSTEM_PROMPT
        assert "MANDATORY RESPONSE FORMAT" in INVESTIGATION_SYSTEM_PROMPT
        assert "claims" in INVESTIGATION_SYSTEM_PROMPT

    def test_user_prompt_formatting(self):
        prompt = build_investigation_user_prompt(
            question="What is CVE-2021-44228?",
            evidence_context="[CVE-001] Log4Shell details...",
        )
        assert "INVESTIGATION QUESTION:" in prompt
        assert "What is CVE-2021-44228?" in prompt
        assert "[CVE-001] Log4Shell details..." in prompt


class TestInvestigationOutputParser:
    def test_parse_valid_json(self):
        raw = """
        {
          "answer": "Evidence [EVTX-001] proves credential dumping occurred.",
          "claims": [
            {
              "claim_id": "CLAIM-001",
              "text": "Credential dumping detected in LSASS process.",
              "supporting_evidence_ids": ["EVTX-001"]
            }
          ],
          "attck_techniques": ["T1003"],
          "cves": [],
          "confidence": "preliminary",
          "limitations": ["Phase 4 verification pending."]
        }
        """
        ans = InvestigationOutputParser.parse(raw, model_name="test-model")
        assert ans.is_parsed_json
        assert "credential dumping" in ans.answer
        assert len(ans.claims) == 1
        assert ans.claims[0].claim_id == "CLAIM-001"
        assert ans.claims[0].supporting_evidence_ids == ["EVTX-001"]
        assert ans.attck_techniques == ["T1003"]
        assert "EVTX-001" in ans.evidence_citations

    def test_parse_markdown_wrapped_json(self):
        raw = """Here is my forensic analysis:
```json
{
  "answer": "PowerShell bypass observed in [DOCX-001] and [IMG-002].",
  "claims": [
    {
      "claim_id": "CLAIM-001",
      "text": "PowerShell execution bypass confirmed.",
      "supporting_evidence_ids": ["DOCX-001", "IMG-002"]
    }
  ],
  "attck_techniques": ["T1059.001"],
  "cves": [],
  "confidence": "moderate",
  "limitations": []
}
```
Investigation complete.
"""
        ans = InvestigationOutputParser.parse(raw)
        assert ans.is_parsed_json
        assert len(ans.claims) == 1
        assert ans.claims[0].supporting_evidence_ids == ["DOCX-001", "IMG-002"]
        assert "DOCX-001" in ans.evidence_citations
        assert "IMG-002" in ans.evidence_citations

    def test_parse_json_with_trailing_comma(self):
        raw = """
        {
          "answer": "Analysis of [EVIDENCE-001].",
          "claims": [
            {
              "claim_id": "CLAIM-001",
              "text": "Finding text",
              "supporting_evidence_ids": ["EVIDENCE-001"],
            }
          ],
          "attck_techniques": ["T1003"],
          "cves": [],
        }
        """
        ans = InvestigationOutputParser.parse(raw)
        assert ans.is_parsed_json
        assert len(ans.claims) == 1
        assert ans.claims[0].supporting_evidence_ids == ["EVIDENCE-001"]

    def test_parse_unstructured_text_fallback(self):
        raw = (
            "Based on the evidence in [DOCX-001], PowerShell was executed with bypass. "
            "This corresponds to technique T1059.001 and vulnerability CVE-2021-44228. "
            "Additional indicators in [EVTX-002] confirm this."
        )
        ans = InvestigationOutputParser.parse(raw)
        assert not ans.is_parsed_json
        assert "DOCX-001" in ans.evidence_citations
        assert "EVTX-002" in ans.evidence_citations
        assert "T1059.001" in ans.attck_techniques
        assert "CVE-2021-44228" in ans.cves
        assert len(ans.claims) > 0

    def test_parse_empty_string(self):
        ans = InvestigationOutputParser.parse("")
        assert not ans.is_parsed_json
        assert ans.confidence == "low"
        assert len(ans.limitations) > 0
