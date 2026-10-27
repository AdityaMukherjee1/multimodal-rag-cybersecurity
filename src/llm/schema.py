"""Data schemas for LLM generation, claims, and structured investigation answers.

Defines the contract between the LLM reasoning layer, the evidence context builder,
and downstream components including the dashboard and future Phase 4 verification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Claim:
    """Individual factual or analytical claim extracted from LLM output.

    Designed for future Phase 4 claim-level evidence verification (NLI/entailment).
    Note: Verification itself is NOT implemented in Phase 3.5.

    Attributes:
        claim_id: Unique identifier for the claim (e.g., 'CLAIM-001').
        text: Specific statement or proposition made by the LLM.
        supporting_evidence_ids: List of canonical evidence IDs cited to justify this claim.
        claim_type: Classification of claim ('factual', 'inferential', 'attribution').
    """

    claim_id: str
    text: str
    supporting_evidence_ids: list[str] = field(default_factory=list)
    claim_type: str = "factual"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Claim:
        return cls(
            claim_id=data.get("claim_id", ""),
            text=data.get("text", ""),
            supporting_evidence_ids=list(data.get("supporting_evidence_ids", [])),
            claim_type=data.get("claim_type", "factual"),
        )


@dataclass
class StructuredInvestigationAnswer:
    """Structured analytical response produced by the LLM reasoning engine.

    Enforces strict grounding, citation of evidence IDs, ATT&CK/CVE attribution,
    and explicit declaration of limitations and epistemic boundaries.

    Attributes:
        answer: Comprehensive narrative answer synthesizing the evidence.
        claims: Disaggregated list of specific claims with supporting evidence IDs.
        attck_techniques: MITRE ATT&CK technique IDs referenced in the evidence.
        cves: Common Vulnerabilities and Exposures IDs referenced in the evidence.
        confidence: Preliminary confidence assessment ('preliminary', 'moderate', 'high').
        limitations: Explicit list of analytical boundaries, gaps, or caveats.
        evidence_citations: Consolidated list of all evidence IDs referenced.
        raw_response: Raw unmodified text returned by the model.
        is_parsed_json: Whether the response was cleanly parsed from structured JSON.
        model_name: Model identifier that generated the response.
        token_usage: Dictionary containing prompt_tokens, completion_tokens, total_tokens.
    """

    answer: str
    claims: list[Claim] = field(default_factory=list)
    attck_techniques: list[str] = field(default_factory=list)
    cves: list[str] = field(default_factory=list)
    confidence: str = "preliminary"
    limitations: list[str] = field(default_factory=list)
    evidence_citations: list[str] = field(default_factory=list)
    raw_response: str = ""
    is_parsed_json: bool = True
    model_name: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["claims"] = [c.to_dict() for c in self.claims]
        return d

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredInvestigationAnswer:
        raw_claims = data.get("claims", [])
        claims = [
            Claim.from_dict(c) if isinstance(c, dict) else Claim(claim_id=f"CLAIM-{i:03d}", text=str(c))
            for i, c in enumerate(raw_claims, start=1)
        ]
        return cls(
            answer=data.get("answer", ""),
            claims=claims,
            attck_techniques=list(data.get("attck_techniques", [])),
            cves=list(data.get("cves", [])),
            confidence=data.get("confidence", "preliminary"),
            limitations=list(data.get("limitations", [])),
            evidence_citations=list(data.get("evidence_citations", [])),
            raw_response=data.get("raw_response", ""),
            is_parsed_json=data.get("is_parsed_json", True),
            model_name=data.get("model_name", ""),
            token_usage=dict(data.get("token_usage", {})),
        )


@dataclass
class LLMResponse:
    """Unified container for an LLM completion including execution metadata."""

    text: str
    structured: StructuredInvestigationAnswer | None = None
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None
