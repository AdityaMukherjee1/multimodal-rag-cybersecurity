"""Benchmark evaluation framework for Phase 4 Claim Verification.

Loads curated cybersecurity ground-truth test cases and measures
verification accuracy, hallucination detection precision, and epistemic reliability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.llm.schema import Claim
from src.multimodal.multimodal_retriever import FusedRetrievalResult
from src.verification.base_verifier import BaseEvidenceVerifier
from src.verification.evidence_verifier import HybridEvidenceVerifier
from src.verification.schema import VerificationStatus

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "evaluation" / "verification_benchmark.json"


def load_verification_benchmark(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load curated benchmark test cases from JSON."""
    p = Path(path) if path else DEFAULT_BENCHMARK_PATH
    if not p.exists():
        raise FileNotFoundError(f"Verification benchmark dataset not found at {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    return data.get("test_cases", [])


def evaluate_verifier_on_benchmark(
    verifier: BaseEvidenceVerifier | None = None,
    mock_evidence_store: dict[str, FusedRetrievalResult] | None = None,
) -> dict[str, Any]:
    """Execute evaluation benchmark and compute performance metrics.

    Args:
        verifier: BaseEvidenceVerifier instance (defaults to HybridEvidenceVerifier).
        mock_evidence_store: Optional pre-loaded dictionary mapping evidence_id to FusedRetrievalResult.

    Returns:
        Dictionary with accuracy, hallucination detection rate, and per-case results.
    """
    v = verifier or HybridEvidenceVerifier()
    test_cases = load_verification_benchmark()

    results = []
    correct_count = 0
    hallucination_detected_count = 0
    total_hallucinations = 0

    for tc in test_cases:
        tc_id = tc["test_id"]
        claim_dict = tc["claim"]
        claim = Claim(
            claim_id=claim_dict["claim_id"],
            text=claim_dict["text"],
            supporting_evidence_ids=claim_dict["supporting_evidence_ids"],
        )
        expected_status = VerificationStatus(tc["expected_status"])

        # Build mock or retrieved evidence for this test case
        evidence_items = []
        for eid in claim.supporting_evidence_ids:
            if mock_evidence_store and eid in mock_evidence_store:
                evidence_items.append(mock_evidence_store[eid])
            else:
                # Synthesize standard canonical evidence item for benchmark
                evidence_items.append(_get_benchmark_evidence_item(eid, tc["query"]))

        verif_res = v.verify_claim(claim=claim, evidence=evidence_items)

        # Check correctness
        is_exact_match = verif_res.status == expected_status
        # In conservative evaluation, UNSUPPORTED and INSUFFICIENT_EVIDENCE are both valid rejection statuses
        is_acceptable_match = is_exact_match or (
            expected_status in (VerificationStatus.UNSUPPORTED, VerificationStatus.INSUFFICIENT_EVIDENCE)
            and verif_res.status in (VerificationStatus.UNSUPPORTED, VerificationStatus.INSUFFICIENT_EVIDENCE)
        )

        if is_acceptable_match:
            correct_count += 1

        is_ground_truth_hallucination = expected_status in (
            VerificationStatus.UNSUPPORTED,
            VerificationStatus.CONTRADICTED,
            VerificationStatus.INSUFFICIENT_EVIDENCE,
        )
        if is_ground_truth_hallucination:
            total_hallucinations += 1
            if verif_res.status in (VerificationStatus.UNSUPPORTED, VerificationStatus.CONTRADICTED, VerificationStatus.INSUFFICIENT_EVIDENCE):
                hallucination_detected_count += 1

        results.append({
            "test_id": tc_id,
            "category": tc["category"],
            "claim_text": claim.text,
            "expected_status": expected_status.value,
            "actual_status": verif_res.status.value,
            "confidence": verif_res.confidence_score,
            "reasoning": verif_res.reasoning,
            "is_correct": is_acceptable_match,
        })

    accuracy = round(correct_count / len(test_cases), 4) if test_cases else 0.0
    hallucination_detection_rate = round(hallucination_detected_count / total_hallucinations, 4) if total_hallucinations else 1.0

    return {
        "verifier": v.verifier_name,
        "total_test_cases": len(test_cases),
        "correct_predictions": correct_count,
        "accuracy": accuracy,
        "hallucination_detection_rate": hallucination_detection_rate,
        "results": results,
    }


def _get_benchmark_evidence_item(evidence_id: str, query: str) -> FusedRetrievalResult:
    """Generate representative canonical evidence record for benchmark evaluation."""
    eid = evidence_id.upper()

    if "T1003" in eid or "SECRET" in eid or "MORDOR" in eid:
        return FusedRetrievalResult(
            evidence_id=evidence_id,
            source_file="T1003-1_Sysmon_CredentialDumping.evtx",
            source_dataset="atomic_evtx",
            modality="evtx",
            content="TargetImage: C:\\Windows\\System32\\lsass.exe. GrantedAccess: 0x1010 (LSASS memory read access). SourceImage: powershell.exe. Credential dumping detected on Local Security Authority Subsystem Service. MITRE ATT&CK T1003.",
            raw_reference="data/raw/atomic_evtx/T1003-1_Sysmon_CredentialDumping.evtx",
            metadata={"attck_technique": "T1003", "event_type": "ProcessAccess"},
            fused_score=0.85,
            rank=1,
            evidence_nature="OBSERVED_EVIDENCE",
        )
    elif "IMG-TERMINAL" in eid:
        return FusedRetrievalResult(
            evidence_id=evidence_id,
            source_file="terminal_credential_access_mimikatz.png",
            source_dataset="sample_images",
            modality="image_screenshot",
            content="privilege::debug sekurlsa::logonpasswords Mimikatz x64",
            raw_reference="data/raw/images/screenshots/terminal_credential_access_mimikatz.png",
            metadata={
                "ocr_text": "mimikatz # privilege::debug\nPrivilege '20' OK\nmimikatz # sekurlsa::logonpasswords\nAuthentication Id : 0 ; 996 (00000000:000003e4)\nSession : Interactive from 1\nUser Name : Administrator",
                "attck_technique": "T1003.001",
            },
            fused_score=0.75,
            rank=1,
            evidence_nature="OBSERVED_EVIDENCE",
        )
    elif "T1059" in eid or "DOCX" in eid:
        return FusedRetrievalResult(
            evidence_id=evidence_id,
            source_file="soc_incident_investigation_briefing.docx",
            source_dataset="sample_docx",
            modality="docx",
            content="Host WORKSTATION-01 spawned powershell.exe with -ExecutionPolicy Bypass. PID 4528. MITRE ATT&CK T1059.001.",
            raw_reference="data/sample/docx/soc_incident_investigation_briefing.docx",
            metadata={"attck_technique": "T1059.001", "event_type": "ProcessCreate"},
            fused_score=0.90,
            rank=1,
            evidence_nature="OBSERVED_EVIDENCE",
        )
    elif "CVE" in eid or "44228" in eid:
        return FusedRetrievalResult(
            evidence_id=evidence_id,
            source_file="nvd_cve_sample.json",
            source_dataset="nvd",
            modality="nvd_cve",
            content="CVE-2021-44228: Apache Log4j2 JNDI features do not protect against attacker controlled LDAP/RMI requests. CVSS 10.0.",
            raw_reference="data/raw/nvd/nvd_cve_sample.json",
            metadata={"cve_id": "CVE-2021-44228"},
            fused_score=0.95,
            rank=1,
            evidence_nature="OBSERVED_EVIDENCE",
        )
    else:
        return FusedRetrievalResult(
            evidence_id=evidence_id,
            source_file="generic_evidence.txt",
            source_dataset="general",
            modality="text",
            content=f"Forensic record {evidence_id} associated with investigation query: {query}",
            raw_reference=f"data/sample/{evidence_id}.txt",
            metadata={},
            fused_score=0.50,
            rank=1,
            evidence_nature="OBSERVED_EVIDENCE",
        )
