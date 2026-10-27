"""Data schemas for Phase 4: Claim-Level Evidence Verification.

Defines the verification status taxonomy, verification result container,
and aggregated verification report structures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class VerificationStatus(str, Enum):
    """5-tier classification hierarchy for claim-evidence entailment."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class ClaimVerificationResult:
    """Individual verification assessment for an extracted claim.

    Attributes:
        claim_id: Unique claim identifier (e.g. 'CLAIM-001').
        claim_text: The specific proposition evaluated.
        supporting_evidence_ids: Evidence IDs cited by the LLM to justify this claim.
        status: Entailment classification (SUPPORTED, PARTIALLY_SUPPORTED, etc.).
        confidence_score: Assessment confidence (0.0 to 1.0).
        reasoning: Verifier explanation detailing why this status was assigned.
        matched_evidence_snippets: Exact text/OCR excerpts from retrieved evidence.
        modality: Modality of the supporting evidence ('text', 'image_screenshot', 'multimodal').
        is_observed_fact: True if grounded in Observed Telemetry/OCR; False if dependent on synthetic VLM captions.
        epistemic_warning: Warning note if claim relies on model-generated interpretation.
        provenance_sources: Disk paths and source files of verified evidence.
    """

    claim_id: str
    claim_text: str
    supporting_evidence_ids: list[str] = field(default_factory=list)
    status: VerificationStatus = VerificationStatus.INSUFFICIENT_EVIDENCE
    confidence_score: float = 0.0
    reasoning: str = ""
    matched_evidence_snippets: list[str] = field(default_factory=list)
    modality: str = "text"
    is_observed_fact: bool = True
    epistemic_warning: str | None = None
    provenance_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaimVerificationResult:
        status_val = data.get("status", "INSUFFICIENT_EVIDENCE")
        if isinstance(status_val, str):
            status = VerificationStatus(status_val)
        else:
            status = status_val

        return cls(
            claim_id=data.get("claim_id", ""),
            claim_text=data.get("claim_text", ""),
            supporting_evidence_ids=list(data.get("supporting_evidence_ids", [])),
            status=status,
            confidence_score=float(data.get("confidence_score", 0.0)),
            reasoning=data.get("reasoning", ""),
            matched_evidence_snippets=list(data.get("matched_evidence_snippets", [])),
            modality=data.get("modality", "text"),
            is_observed_fact=bool(data.get("is_observed_fact", True)),
            epistemic_warning=data.get("epistemic_warning"),
            provenance_sources=list(data.get("provenance_sources", [])),
        )


@dataclass
class VerificationReport:
    """Comprehensive verification summary for an investigation query.

    Attributes:
        query: The user query being investigated.
        claim_results: List of individual claim verification outcomes.
        total_claims: Total count of evaluated claims.
        supported_count: Number of fully supported claims.
        partially_supported_count: Number of partially supported claims.
        unsupported_count: Number of unsupported claims.
        contradicted_count: Number of contradicted claims.
        insufficient_evidence_count: Number of claims with insufficient evidence.
        hallucination_rate: Proportion of claims that are unsupported or contradicted.
        overall_grounding_score: Composite grounding metric (0.0 to 1.0).
        verification_latency_ms: Execution duration in milliseconds.
        verifier_name: Identifier of the verifier component.
        verified_answer: Synthesis narrative integrating verification assessments.
    """

    query: str
    claim_results: list[ClaimVerificationResult] = field(default_factory=list)
    total_claims: int = 0
    supported_count: int = 0
    partially_supported_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    insufficient_evidence_count: int = 0
    hallucination_rate: float = 0.0
    overall_grounding_score: float = 0.0
    verification_latency_ms: float = 0.0
    verifier_name: str = "HybridEvidenceVerifier"
    verified_answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["claim_results"] = [c.to_dict() for c in self.claim_results]
        return d

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationReport:
        raw_results = data.get("claim_results", [])
        claim_results = [
            ClaimVerificationResult.from_dict(c) if isinstance(c, dict) else c
            for c in raw_results
        ]
        return cls(
            query=data.get("query", ""),
            claim_results=claim_results,
            total_claims=int(data.get("total_claims", len(claim_results))),
            supported_count=int(data.get("supported_count", 0)),
            partially_supported_count=int(data.get("partially_supported_count", 0)),
            unsupported_count=int(data.get("unsupported_count", 0)),
            contradicted_count=int(data.get("contradicted_count", 0)),
            insufficient_evidence_count=int(data.get("insufficient_evidence_count", 0)),
            hallucination_rate=float(data.get("hallucination_rate", 0.0)),
            overall_grounding_score=float(data.get("overall_grounding_score", 0.0)),
            verification_latency_ms=float(data.get("verification_latency_ms", 0.0)),
            verifier_name=data.get("verifier_name", "HybridEvidenceVerifier"),
            verified_answer=data.get("verified_answer", ""),
        )
