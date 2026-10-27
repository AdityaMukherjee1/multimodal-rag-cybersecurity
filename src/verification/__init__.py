"""Phase 4 Claim Extraction and Evidence Verification subsystem.

Provides claim-level entailment evaluation, epistemic modality audits,
and hallucination reduction metrics for cybersecurity investigations.
"""

from src.verification.base_verifier import BaseEvidenceVerifier
from src.verification.evidence_verifier import HybridEvidenceVerifier
from src.verification.llm_verifier import LLMEvidenceVerifier
from src.verification.schema import ClaimVerificationResult, VerificationReport, VerificationStatus
from src.verification.verification_benchmark import evaluate_verifier_on_benchmark, load_verification_benchmark

__all__ = [
    "VerificationStatus",
    "ClaimVerificationResult",
    "VerificationReport",
    "BaseEvidenceVerifier",
    "HybridEvidenceVerifier",
    "LLMEvidenceVerifier",
    "load_verification_benchmark",
    "evaluate_verifier_on_benchmark",
]
