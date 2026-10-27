"""Abstract base class for evidence verifiers.

Defines the interface that all verification strategies must implement,
enabling comparison between rule-based, semantic, hybrid, and LLM-based verifiers.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from src.llm.schema import Claim
from src.multimodal.multimodal_retriever import FusedRetrievalResult
from src.verification.schema import ClaimVerificationResult, VerificationReport, VerificationStatus


class BaseEvidenceVerifier(ABC):
    """Abstract interface for claim verification implementations."""

    @abstractmethod
    def verify_claim(
        self,
        claim: Claim,
        evidence: list[FusedRetrievalResult],
    ) -> ClaimVerificationResult:
        """Verify an individual claim against candidate evidence.

        Args:
            claim: Extracted Claim dataclass.
            evidence: Top-K retrieved FusedRetrievalResult objects.

        Returns:
            ClaimVerificationResult with status, confidence, and reasoning.
        """
        ...

    def verify_all(
        self,
        claims: list[Claim],
        evidence: list[FusedRetrievalResult],
        query: str = "",
        raw_answer: str = "",
    ) -> VerificationReport:
        """Verify all extracted claims and compute aggregated metrics.

        Args:
            claims: List of extracted claims.
            evidence: Top-K retrieved evidence candidates.
            query: Original investigation query.
            raw_answer: Unverified narrative answer from the LLM.

        Returns:
            VerificationReport containing detailed results and grounding metrics.
        """
        start_t = time.perf_counter()

        if not claims:
            return VerificationReport(
                query=query,
                claim_results=[],
                total_claims=0,
                verifier_name=self.verifier_name,
                verification_latency_ms=0.0,
                verified_answer=raw_answer or "No discrete claims were available for verification.",
            )

        results: list[ClaimVerificationResult] = []
        for claim in claims:
            res = self.verify_claim(claim=claim, evidence=evidence)
            results.append(res)

        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

        # Calculate counts
        supp_n = sum(1 for r in results if r.status == VerificationStatus.SUPPORTED)
        part_n = sum(1 for r in results if r.status == VerificationStatus.PARTIALLY_SUPPORTED)
        unsupp_n = sum(1 for r in results if r.status == VerificationStatus.UNSUPPORTED)
        contra_n = sum(1 for r in results if r.status == VerificationStatus.CONTRADICTED)
        insuff_n = sum(1 for r in results if r.status == VerificationStatus.INSUFFICIENT_EVIDENCE)
        total_n = len(results)

        # Hallucination rate = (unsupported + contradicted) / total
        hallucination_rate = round((unsupp_n + contra_n) / total_n, 4) if total_n > 0 else 0.0

        # Composite grounding score: (1.0*supp + 0.5*part + 0.1*insuff - 1.0*contra) / total
        raw_score = (1.0 * supp_n + 0.5 * part_n + 0.1 * insuff_n - 1.0 * contra_n) / total_n if total_n > 0 else 0.0
        grounding_score = round(max(0.0, min(1.0, raw_score)), 4)

        # Construct verified narrative
        verified_answer = self._generate_verified_answer_narrative(raw_answer, results)

        return VerificationReport(
            query=query,
            claim_results=results,
            total_claims=total_n,
            supported_count=supp_n,
            partially_supported_count=part_n,
            unsupported_count=unsupp_n,
            contradicted_count=contra_n,
            insufficient_evidence_count=insuff_n,
            hallucination_rate=hallucination_rate,
            overall_grounding_score=grounding_score,
            verification_latency_ms=latency_ms,
            verifier_name=self.verifier_name,
            verified_answer=verified_answer,
        )

    def _generate_verified_answer_narrative(
        self,
        raw_answer: str,
        results: list[ClaimVerificationResult],
    ) -> str:
        """Synthesize verified answer narrative highlighting confirmed vs unverified claims."""
        lines: list[str] = []

        if raw_answer:
            lines.append("### Original Finding (Post-Verification Audit)")
            lines.append(raw_answer)
            lines.append("")

        lines.append("### Verified Evidence Audit Summary")
        for r in results:
            badge = f"[{r.status.value}]"
            lines.append(f"- **{r.claim_id}** {badge}: {r.claim_text}")
            if r.status != VerificationStatus.SUPPORTED and r.reasoning:
                lines.append(f"  *Note: {r.reasoning}*")
            if r.epistemic_warning:
                lines.append(f"  *Epistemic Warning: {r.epistemic_warning}*")

        return "\n".join(lines)

    @property
    def verifier_name(self) -> str:
        """Return identifier of this verifier."""
        return self.__class__.__name__
