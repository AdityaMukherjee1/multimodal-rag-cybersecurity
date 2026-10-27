"""LLM-based Evidence Verifier (Critic / Referee).

Uses an independent LLM verification prompt to judge natural language inference
and entailment between retrieved evidence and generated claims.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.llm.base_llm import BaseLLM
from src.llm.llm_client import create_llm
from src.llm.output_parser import InvestigationOutputParser
from src.llm.schema import Claim
from src.multimodal.multimodal_retriever import FusedRetrievalResult
from src.verification.base_verifier import BaseEvidenceVerifier
from src.verification.evidence_verifier import HybridEvidenceVerifier
from src.verification.schema import ClaimVerificationResult, VerificationStatus

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are an independent, conservative cybersecurity forensic evidence verifier.

Your task is to judge whether a given CLAIM is strictly supported by the provided PREMISE (Retrieved Forensic Evidence).

### STRICT VERIFICATION RULES:
1. SUPPORTED: The claim is directly and factually proven by the retrieved evidence.
2. PARTIALLY_SUPPORTED: The claim is partially substantiated, but makes additional unverified assertions or relies on synthetic model interpretations.
3. UNSUPPORTED: The claim makes assertions that have no basis in the retrieved evidence.
4. CONTRADICTED: The claim directly conflicts with or is refuted by the retrieved evidence.
5. INSUFFICIENT_EVIDENCE: The evidence is ambiguous, absent, or incomplete to substantiate the claim.

You MUST respond with valid JSON matching this schema:
{
  "status": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
  "confidence_score": 0.0 to 1.0,
  "reasoning": "Clear explanation of the verdict.",
  "matched_snippets": ["Key evidence quote 1"],
  "is_observed_fact": true | false
}
"""


class LLMEvidenceVerifier(BaseEvidenceVerifier):
    """Verifier using an LLM critic for natural language inference."""

    def __init__(
        self,
        llm_provider: BaseLLM | None = None,
        provider_name: str = "mock",
        fallback_verifier: BaseEvidenceVerifier | None = None,
    ) -> None:
        self.llm = llm_provider or create_llm(provider_name)
        self.fallback = fallback_verifier or HybridEvidenceVerifier()

    def verify_claim(
        self,
        claim: Claim,
        evidence: list[FusedRetrievalResult],
    ) -> ClaimVerificationResult:
        """Verify claim using LLM referee prompt with fallback."""
        if not evidence:
            return self.fallback.verify_claim(claim, evidence)

        # Prepare premise context
        premise_lines = []
        for e in evidence:
            premise_lines.append(f"[{e.evidence_id}] ({e.modality}): {e.content[:600]}")
        premise_str = "\n\n".join(premise_lines)

        user_prompt = f"""EVIDENCE PREMISE:
{premise_str}

HYPOTHESIS CLAIM TO VERIFY:
Claim ID: {claim.claim_id}
Claim Text: "{claim.text}"
Cited Evidence IDs: {claim.supporting_evidence_ids}

Evaluate whether the evidence premise entails this claim. Respond strictly in the required JSON format.
"""

        try:
            raw_res = self.llm.generate(
                prompt=user_prompt,
                system_prompt=VERIFIER_SYSTEM_PROMPT,
            )

            parsed = self._parse_verifier_json(raw_res)
            if parsed is not None:
                status_str = parsed.get("status", "INSUFFICIENT_EVIDENCE").upper()
                try:
                    status = VerificationStatus(status_str)
                except ValueError:
                    status = VerificationStatus.INSUFFICIENT_EVIDENCE

                return ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    supporting_evidence_ids=claim.supporting_evidence_ids,
                    status=status,
                    confidence_score=float(parsed.get("confidence_score", 0.7)),
                    reasoning=str(parsed.get("reasoning", "LLM verification evaluation")),
                    matched_evidence_snippets=list(parsed.get("matched_snippets", [])),
                    modality="multimodal" if any("image" in e.modality for e in evidence) else "text",
                    is_observed_fact=bool(parsed.get("is_observed_fact", True)),
                    provenance_sources=[e.raw_reference for e in evidence if e.raw_reference],
                )

        except Exception as exc:
            logger.warning("LLMEvidenceVerifier failed: %s. Falling back to HybridEvidenceVerifier.", exc)

        # Fallback to deterministic hybrid verifier
        return self.fallback.verify_claim(claim, evidence)

    def _parse_verifier_json(self, raw_text: str) -> dict[str, Any] | None:
        """Extract JSON from LLM output."""
        try:
            # Handle markdown wrapping
            text = raw_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            if isinstance(data, dict) and "status" in data:
                return data
        except Exception:
            pass
        return None
