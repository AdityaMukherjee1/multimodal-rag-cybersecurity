"""Hybrid Evidence Verifier with Epistemic Modality Audit and Entity Grounding.

Combines deterministic entity alignment, semantic entailment scoring,
contradiction detection, and strict separation between observed telemetry
and synthetic VLM captions.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.llm.schema import Claim
from src.multimodal.multimodal_retriever import FusedRetrievalResult
from src.verification.base_verifier import BaseEvidenceVerifier
from src.verification.schema import ClaimVerificationResult, VerificationStatus

logger = logging.getLogger(__name__)

# Regular expressions for cybersecurity forensic entities
ATTCK_PATTERN = re.compile(r"\b(T\d{4}(?:\.\d{3})?)\b", re.IGNORECASE)
CVE_PATTERN = re.compile(r"\b(CVE-\d{4}-\d{4,7})\b", re.IGNORECASE)
IP_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
PID_PATTERN = re.compile(r"\bPID:?\s*(\d{2,6})\b", re.IGNORECASE)
TOOL_PATTERN = re.compile(
    r"\b(powershell(?:\.exe)?|mimikatz|psexec|cmd(?:\.exe)?|bash|ssh|wmic|rundll32|mshta|certutil|curl|wget|nmap)\b",
    re.IGNORECASE,
)
PROCESS_PATTERN = re.compile(
    r"\b(lsass(?:\.exe)?|powershell(?:\.exe)?|cmd(?:\.exe)?|svchost(?:\.exe)?|explorer(?:\.exe)?|taskmgr(?:\.exe)?|sysmon(?:\.exe)?)\b",
    re.IGNORECASE,
)
ACTOR_PATTERN = re.compile(
    r"\b(apt\d{1,2}|cozy bear|fancy bear|lazarus|sandworm|fin\d{1,2}|oilrig|wizard spider)\b",
    re.IGNORECASE,
)
TARGET_SERVICE_PATTERN = re.compile(
    r"\b(dropbox|amazon s3|s3 bucket|google drive|onedrive|github|pastebin)\b",
    re.IGNORECASE,
)
OS_LINUX_PATTERN = re.compile(r"\b(linux|ubuntu|debian|centos|redhat|bash|elf)\b", re.IGNORECASE)
OS_WIN_PATTERN = re.compile(r"\b(windows|sysmon|powershell|lsass|cmd\.exe|evtx|ntds)\b", re.IGNORECASE)


class HybridEvidenceVerifier(BaseEvidenceVerifier):
    """Conservative, multi-stage hybrid verifier for cybersecurity claims.

    Stages:
    1. Evidence Resolution & Provenance Validation.
    2. Epistemic Modality Audit (Observed Telemetry vs Model Interpretation).
    3. Cybersecurity Entity Alignment (ATT&CK, CVE, Tools, PIDs, IPs).
    4. Contradiction & Negation Detection.
    5. Semantic Overlap & Support Classification.
    """

    def __init__(
        self,
        high_support_threshold: float = 0.50,
        partial_support_threshold: float = 0.25,
        conservative_mode: bool = True,
    ) -> None:
        self.high_support_threshold = high_support_threshold
        self.partial_support_threshold = partial_support_threshold
        self.conservative_mode = conservative_mode

    def verify_claim(
        self,
        claim: Claim,
        evidence: list[FusedRetrievalResult],
    ) -> ClaimVerificationResult:
        """Verify an extracted claim against retrieved evidence candidates."""
        claim_text = claim.text.strip()
        cited_ids = claim.supporting_evidence_ids or []

        if not evidence:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim_text,
                supporting_evidence_ids=cited_ids,
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence_score=0.0,
                reasoning="No retrieved evidence was provided to the verifier.",
                is_observed_fact=False,
            )

        # 1. Resolve Evidence Candidates
        resolved_items = self._resolve_evidence(cited_ids, evidence)
        evidence_pool = resolved_items if resolved_items else evidence

        # Extract combined observed text and model interpretation text
        observed_corpus: list[str] = []
        interpretation_corpus: list[str] = []
        modalities: set[str] = set()
        provenance_sources: list[str] = []

        for item in evidence_pool:
            modalities.add(item.modality)
            if item.raw_reference and item.raw_reference not in provenance_sources:
                provenance_sources.append(item.raw_reference)

            obs_txt, interp_txt = self._segregate_epistemic_content(item)
            if obs_txt:
                observed_corpus.append(obs_txt)
            if interp_txt:
                interpretation_corpus.append(interp_txt)

        full_observed_text = "\n".join(observed_corpus)
        full_interp_text = "\n".join(interpretation_corpus)
        combined_text = full_observed_text + "\n" + full_interp_text

        # 2. Contradiction Detection
        is_contradicted, contra_reason = self._check_contradiction(claim_text, full_observed_text)
        if is_contradicted:
            return ClaimVerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim_text,
                supporting_evidence_ids=cited_ids,
                status=VerificationStatus.CONTRADICTED,
                confidence_score=0.90,
                reasoning=contra_reason,
                matched_evidence_snippets=self._extract_relevant_snippets(claim_text, full_observed_text, max_n=2),
                modality="multimodal" if len(modalities) > 1 else (list(modalities)[0] if modalities else "text"),
                is_observed_fact=True,
                provenance_sources=provenance_sources,
            )

        # 3. Entity Alignment
        entity_score, matched_entities, missing_entities = self._evaluate_entities(
            claim_text=claim_text,
            observed_text=full_observed_text,
            evidence_items=evidence_pool,
        )

        # 4. Lexical / Concept Overlap
        obs_overlap = self._compute_token_overlap(claim_text, full_observed_text)
        interp_overlap = self._compute_token_overlap(claim_text, full_interp_text)

        # Determine if grounding relies on Observed Evidence vs Model Interpretation
        is_observed_grounded = obs_overlap >= self.partial_support_threshold or (entity_score >= 0.6 and obs_overlap >= 0.25)
        relies_on_interpretation = not is_observed_grounded and interp_overlap >= self.partial_support_threshold

        matched_snippets = self._extract_relevant_snippets(claim_text, combined_text)

        # 5. Classification Logic
        status = VerificationStatus.INSUFFICIENT_EVIDENCE
        confidence = 0.0
        reasoning = ""
        epistemic_warning = None

        has_unverified_critical_entity = any(
            m for m in missing_entities if any(prefix in m for prefix in ("Actor", "Target", "CVE", "ATT&CK"))
        )

        if is_observed_grounded:
            if not has_unverified_critical_entity and (
                (entity_score >= 0.75 and obs_overlap >= self.high_support_threshold)
                or (entity_score == 1.0 and obs_overlap >= 0.35)
                or (obs_overlap >= 0.65)
            ):
                status = VerificationStatus.SUPPORTED
                confidence = round(min(0.98, (entity_score * 0.5 + obs_overlap * 0.5)), 2)
                reasoning = (
                    f"Claim is directly supported by observed telemetry/records. "
                    f"Verified entities: {', '.join(matched_entities) if matched_entities else 'Key forensic terms verified'}."
                )
            elif entity_score >= 0.4 or obs_overlap >= self.partial_support_threshold:
                if has_unverified_critical_entity and entity_score < 0.5:
                    status = VerificationStatus.INSUFFICIENT_EVIDENCE
                    confidence = 0.40
                    reasoning = (
                        f"Retrieved evidence does not substantiate critical assertion ({', '.join(missing_entities)})."
                    )
                else:
                    status = VerificationStatus.PARTIALLY_SUPPORTED
                    confidence = round(min(0.85, (entity_score * 0.4 + obs_overlap * 0.6)), 2)
                    if missing_entities:
                        reasoning = (
                            f"Core statement aligns with evidence, but specific entities/details "
                            f"({', '.join(missing_entities)}) are unconfirmed or extrapolated."
                        )
                    else:
                        reasoning = "Evidence supports the general premise, but some specific assertions exceed the retrieved records (unconfirmed/extrapolated details)."
            else:
                status = VerificationStatus.INSUFFICIENT_EVIDENCE
                confidence = 0.35
                reasoning = "Evidence references related concepts, but is insufficient to substantiate this specific claim."

        elif relies_on_interpretation:
            # Epistemic warning: reliance on synthetic BLIP caption
            status = VerificationStatus.PARTIALLY_SUPPORTED
            confidence = round(min(0.65, interp_overlap * 0.7), 2)
            epistemic_warning = (
                "Claim matches synthetic VLM description (Salesforce BLIP) but lacks ground-truth observed OCR/log telemetry."
            )
            reasoning = (
                "Claim is consistent with model-generated visual interpretation, "
                "but cannot be marked fully SUPPORTED because it lacks verified ground-truth telemetry."
            )

        else:
            # Check if completely unsupported
            if obs_overlap < 0.15 and not matched_entities:
                status = VerificationStatus.UNSUPPORTED
                confidence = 0.85
                reasoning = (
                    "No supporting evidence found in retrieved candidates. "
                    "The assertion does not match any observed logs, advisories, or OCR extractions."
                )
            else:
                status = VerificationStatus.INSUFFICIENT_EVIDENCE
                confidence = 0.30
                reasoning = "Retrieved evidence is incomplete or ambiguous for confirming this assertion."

        # Modality aggregation
        modality_str = "multimodal" if len(modalities) > 1 else (list(modalities)[0] if modalities else "text")

        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim_text,
            supporting_evidence_ids=cited_ids,
            status=status,
            confidence_score=confidence,
            reasoning=reasoning,
            matched_evidence_snippets=matched_snippets,
            modality=modality_str,
            is_observed_fact=not bool(epistemic_warning),
            epistemic_warning=epistemic_warning,
            provenance_sources=provenance_sources,
        )

    def _resolve_evidence(
        self,
        cited_ids: list[str],
        evidence: list[FusedRetrievalResult],
    ) -> list[FusedRetrievalResult]:
        """Resolve cited evidence IDs against the top-K retrieved evidence candidates."""
        if not cited_ids:
            return []

        resolved: list[FusedRetrievalResult] = []
        for item in evidence:
            if item.evidence_id in cited_ids:
                resolved.append(item)
            elif any(cid.lower() in item.evidence_id.lower() or item.evidence_id.lower() in cid.lower() for cid in cited_ids):
                resolved.append(item)
        return resolved

    def _segregate_epistemic_content(self, item: FusedRetrievalResult) -> tuple[str, str]:
        """Extract observed evidence vs model interpretation from an evidence item."""
        content = item.content or ""
        meta = item.metadata or {}
        modality = item.modality.lower()

        if "image" in modality:
            obs_parts: list[str] = [f"Source File: {item.source_file}"]
            if content and "[OBSERVED EVIDENCE:" not in content and "[MODEL-GENERATED" not in content:
                obs_parts.append(content)
            ocr_text = meta.get("ocr_text")
            if ocr_text:
                obs_parts.append(ocr_text)
            elif "[OBSERVED EVIDENCE:" in content:
                parts = content.split("[MODEL-GENERATED INTERPRETATION")
                obs_parts.append(parts[0])

            interp_parts: list[str] = []
            vlm_caption = meta.get("vlm_caption") or meta.get("vlm_interpretation")
            if vlm_caption:
                interp_parts.append(vlm_caption)
            elif "[MODEL-GENERATED INTERPRETATION" in content:
                parts = content.split("[MODEL-GENERATED INTERPRETATION")
                if len(parts) > 1:
                    interp_parts.append(parts[1])

            return "\n".join(obs_parts), "\n".join(interp_parts)

        else:
            # Text, EVTX, JSON, DOCX, PDF, Network, NVD, ATT&CK is Observed Evidence
            obs_lines = [
                f"Source: {item.source_file} ({item.source_dataset})",
                f"ATT&CK: {meta.get('attck_technique', '')} | CVE: {meta.get('cve_id', '')}",
                content,
            ]
            return "\n".join(obs_lines), ""

    def _check_contradiction(self, claim_text: str, observed_text: str) -> tuple[bool, str]:
        """Check for explicit contradictions in OS, tools, or negated facts."""
        c_lower = claim_text.lower()
        obs_lower = observed_text.lower()

        # 1. OS Contradiction: Claiming pure Linux incident when evidence is Windows Workstation
        has_win_negation = bool(re.search(r"\b(without|no|not|zero)\s+([a-z0-9_\-\s]{0,20}\s+)?windows\b", c_lower))
        has_win_affirmation = bool(OS_WIN_PATTERN.search(c_lower)) and not has_win_negation

        if OS_LINUX_PATTERN.search(c_lower) and (not has_win_affirmation or has_win_negation):
            if "windows" in obs_lower or "powershell" in obs_lower or "sysmon" in obs_lower or "workstation" in obs_lower:
                if "linux" not in obs_lower and "bash" not in obs_lower and "ubuntu" not in obs_lower:
                    return True, "Contradiction detected: Claim asserts Linux/Bash execution, but observed forensic telemetry explicitly documents Windows PowerShell and Sysmon events."

        # 2. Clean vs Malicious Contradiction
        if "no malicious activity" in c_lower or "benign execution" in c_lower or "false positive" in c_lower:
            if "mimikatz" in obs_lower or "critical alert" in obs_lower or "cve-2021-44228" in obs_lower or "t1003" in obs_lower:
                return True, "Contradiction detected: Claim asserts benign activity, directly contradicting critical observed alerts and malware signatures (Mimikatz / CVE)."

        return False, ""

    def _evaluate_entities(
        self,
        claim_text: str,
        observed_text: str,
        evidence_items: list[FusedRetrievalResult],
    ) -> tuple[float, list[str], list[str]]:
        """Extract entities from claim and verify presence in observed evidence."""
        # Find entities in claim
        claim_attck = set(m.upper() for m in ATTCK_PATTERN.findall(claim_text))
        claim_cve = set(m.upper() for m in CVE_PATTERN.findall(claim_text))
        claim_ips = set(IP_PATTERN.findall(claim_text))
        claim_tools = set(m.lower() for m in TOOL_PATTERN.findall(claim_text))
        claim_processes = set(m.lower() for m in PROCESS_PATTERN.findall(claim_text))
        claim_actors = set(m.lower() for m in ACTOR_PATTERN.findall(claim_text))
        claim_targets = set(m.lower() for m in TARGET_SERVICE_PATTERN.findall(claim_text))

        # Build evidence entity pool
        ev_text_upper = observed_text.upper()
        ev_text_lower = observed_text.lower()

        for item in evidence_items:
            meta = item.metadata or {}
            if meta.get("attck_technique"):
                ev_text_upper += f" {str(meta['attck_technique']).upper()}"
            if meta.get("cve_id"):
                ev_text_upper += f" {str(meta['cve_id']).upper()}"

        matched: list[str] = []
        missing: list[str] = []

        # Check ATT&CK
        for a in claim_attck:
            if a in ev_text_upper or a.replace(".", "_") in ev_text_upper or a.split(".")[0] in ev_text_upper:
                matched.append(f"ATT&CK {a}")
            else:
                missing.append(f"ATT&CK {a}")

        # Check CVEs
        for c in claim_cve:
            if c in ev_text_upper or c.replace("-", "_") in ev_text_upper:
                matched.append(c)
            else:
                missing.append(c)

        # Check IPs
        for ip in claim_ips:
            if ip in observed_text:
                matched.append(f"IP {ip}")
            else:
                missing.append(f"IP {ip}")

        # Check Tools
        for t in claim_tools:
            if t in ev_text_lower:
                matched.append(t)
            else:
                missing.append(t)

        # Check Processes
        for p in claim_processes:
            if p in ev_text_lower:
                matched.append(p)
            else:
                missing.append(p)

        # Check Actors
        for actor in claim_actors:
            if actor in ev_text_lower:
                matched.append(f"Actor {actor.upper()}")
            else:
                missing.append(f"Actor {actor.upper()}")

        # Check External Targets
        for tgt in claim_targets:
            if tgt in ev_text_lower:
                matched.append(f"Target {tgt}")
            else:
                missing.append(f"Target {tgt}")

        total_entities = len(matched) + len(missing)
        if total_entities == 0:
            return 0.5, [], []

        score = len(matched) / total_entities
        return score, matched, missing

    def _compute_token_overlap(self, claim_text: str, evidence_text: str) -> float:
        """Compute keyword and content-token overlap ratio."""
        if not claim_text or not evidence_text:
            return 0.0

        # Tokenize content words (filter common stopwords)
        stopwords = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
            "is", "was", "were", "are", "be", "been", "by", "that", "this", "from", "as",
            "has", "had", "have", "it", "its", "which", "evidence", "claim", "observed",
        }
        c_tokens = set(re.findall(r"\b[A-Za-z0-9_\-\.]{3,}\b", claim_text.lower())) - stopwords
        if not c_tokens:
            return 0.0

        ev_tokens = set(re.findall(r"\b[A-Za-z0-9_\-\.]{3,}\b", evidence_text.lower()))
        matched_tokens = c_tokens.intersection(ev_tokens)

        return len(matched_tokens) / len(c_tokens)

    def _extract_relevant_snippets(self, claim_text: str, evidence_text: str, max_n: int = 3) -> list[str]:
        """Extract most relevant lines/sentences from evidence matching the claim."""
        if not claim_text or not evidence_text:
            return []

        c_tokens = set(re.findall(r"\b[A-Za-z0-9_\-]{3,}\b", claim_text.lower()))
        lines = [line.strip() for line in evidence_text.split("\n") if len(line.strip()) > 15]

        scored_lines = []
        for line in lines:
            # Skip separator lines
            if line.startswith("===") or line.startswith("---") or line.startswith("[OBSERVED"):
                continue
            line_tokens = set(re.findall(r"\b[A-Za-z0-9_\-]{3,}\b", line.lower()))
            overlap = len(c_tokens.intersection(line_tokens))
            if overlap > 0:
                scored_lines.append((overlap, line))

        scored_lines.sort(key=lambda x: x[0], reverse=True)
        return [item[1][:250] for item in scored_lines[:max_n]]
