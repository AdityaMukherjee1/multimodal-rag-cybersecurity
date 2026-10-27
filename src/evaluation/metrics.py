"""Evaluation metrics for Phase 5 Empirical Comparative Analysis.

Calculates retrieval metrics (Recall@K, MRR), answer grounding, citation precision,
claim verification metrics (Precision, Recall, F1, Hallucination Detection Rate),
and performance telemetry.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.verification.schema import VerificationReport, VerificationStatus

logger = logging.getLogger(__name__)


def is_evidence_match(candidate_id: str, ground_truth_id: str) -> bool:
    """Determine whether candidate evidence ID matches ground truth reference."""
    c = candidate_id.strip().lower()
    gt = ground_truth_id.strip().lower()

    if c == gt:
        return True
    if c in gt or gt in c:
        return True

    # Strip common file extensions
    c_base = re.sub(r"\.(docx|json|evtx|png|pdf|csv|txt)$", "", c)
    gt_base = re.sub(r"\.(docx|json|evtx|png|pdf|csv|txt)$", "", gt)
    if c_base in gt_base or gt_base in c_base:
        return True

    # Check normalized technique / file / CVE components
    c_norm = c_base.replace("-", "_").replace(".", "_")
    gt_norm = gt_base.replace("-", "_").replace(".", "_")
    if c_norm == gt_norm or c_norm in gt_norm or gt_norm in c_norm:
        return True

    # Check core tokens
    gt_tokens = [t for t in re.split(r"[_\-\.]+", gt_base) if len(t) > 4]
    c_tokens = [t for t in re.split(r"[_\-\.]+", c_base) if len(t) > 4]
    if gt_tokens and c_tokens and any(t in c_tokens for t in gt_tokens):
        return True

    return False


def calculate_retrieval_metrics(
    retrieved_ids: list[str],
    ground_truth_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute Recall@K and Mean Reciprocal Rank (MRR).

    Args:
        retrieved_ids: List of retrieved evidence identifiers in ranked order.
        ground_truth_ids: List of relevant ground truth evidence identifiers.
        k_values: List of cutoffs to evaluate (defaults to [1, 3, 5, 10]).

    Returns:
        Dictionary mapping Recall@K and MRR to float values (0.0 to 1.0).
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    if not ground_truth_ids:
        # Edge case: No ground truth evidence exists (e.g. unsupported adversarial prompt)
        return {f"recall@{k}": 1.0 for k in k_values} | {"mrr": 1.0}

    metrics: dict[str, float] = {}

    for k in k_values:
        top_k = retrieved_ids[:k]
        hits = 0
        for gt in ground_truth_ids:
            if any(is_evidence_match(r, gt) for r in top_k):
                hits += 1
        metrics[f"recall@{k}"] = round(hits / len(ground_truth_ids), 4)

    # Calculate MRR (Reciprocal Rank of first relevant document)
    mrr = 0.0
    for rank_idx, r in enumerate(retrieved_ids, start=1):
        if any(is_evidence_match(r, gt) for gt in ground_truth_ids):
            mrr = 1.0 / rank_idx
            break
    metrics["mrr"] = round(mrr, 4)

    return metrics


def calculate_answer_accuracy(answer_text: str, key_facts: list[str]) -> float:
    """Measure coverage of ground truth reference keywords/facts in the answer."""
    if not key_facts:
        return 1.0
    if not answer_text:
        return 0.0

    ans_lower = answer_text.lower()
    matched = 0.0
    for fact in key_facts:
        f_clean = fact.strip().lower()
        if f_clean in ans_lower:
            matched += 1.0
        else:
            tokens = [t for t in re.split(r"[\s\.\-_]+", f_clean) if len(t) > 3]
            if tokens and any(token in ans_lower for token in tokens):
                matched += 0.5

    return round(min(1.0, matched / len(key_facts)), 4)


def calculate_citation_accuracy(
    cited_ids: list[str],
    retrieved_ids: list[str],
    ground_truth_ids: list[str],
) -> float:
    """Calculate precision of evidence citations in generated claims.

    A citation is valid if it points to a ground-truth evidence item or a retrieved item.
    """
    if not cited_ids:
        return 1.0 if not ground_truth_ids else 0.0

    valid_citations = 0
    for cid in cited_ids:
        is_gt = any(is_evidence_match(cid, gt) for gt in ground_truth_ids)
        is_retrieved = any(is_evidence_match(cid, r) for r in retrieved_ids)
        if is_gt or is_retrieved:
            valid_citations += 1

    return round(valid_citations / len(cited_ids), 4)


def calculate_verification_metrics(
    predicted_statuses: list[VerificationStatus | str],
    expected_statuses: list[VerificationStatus | str],
) -> dict[str, float]:
    """Compute classification accuracy, precision, recall, and F1 for hallucination detection.

    Hallucination class = UNSUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE.
    Grounded class = SUPPORTED, PARTIALLY_SUPPORTED.
    """
    if not predicted_statuses or not expected_statuses:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hallucination_detection_rate": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    total = min(len(predicted_statuses), len(expected_statuses))
    correct = 0

    tp, fp, fn, tn = 0, 0, 0, 0

    hallucination_set = {
        VerificationStatus.UNSUPPORTED.value,
        VerificationStatus.CONTRADICTED.value,
        VerificationStatus.INSUFFICIENT_EVIDENCE.value,
    }

    for i in range(total):
        p_val = predicted_statuses[i].value if hasattr(predicted_statuses[i], "value") else str(predicted_statuses[i])
        e_val = expected_statuses[i].value if hasattr(expected_statuses[i], "value") else str(expected_statuses[i])

        # Exact or conservative match
        if p_val == e_val or (p_val in hallucination_set and e_val in hallucination_set):
            correct += 1

        is_pred_hallucination = p_val in hallucination_set
        is_exp_hallucination = e_val in hallucination_set

        if is_pred_hallucination and is_exp_hallucination:
            tp += 1
        elif is_pred_hallucination and not is_exp_hallucination:
            fp += 1
        elif not is_pred_hallucination and is_exp_hallucination:
            fn += 1
        else:
            tn += 1

    accuracy = round(correct / total, 4) if total > 0 else 0.0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 1.0
    f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0.0
    hallucination_detection_rate = recall

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_detection_rate": hallucination_detection_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
