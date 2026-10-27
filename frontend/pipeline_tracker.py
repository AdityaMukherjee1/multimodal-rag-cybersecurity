"""Execution State & Architecture Tracker for Multimodal RAG Pipeline.

Tracks authentic component states, execution flow, latency breakdown,
and token accounting during investigation query execution.
Phase 3.5: Multimodal RAG + Real LLM Reasoning & Claim Extraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class StepStatus(str, Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class PipelineStep:
    name: str
    description: str
    status: StepStatus = StepStatus.WAITING
    duration_ms: float = 0.0
    details: str = ""


@dataclass
class ExecutionMetrics:
    total_latency_ms: float = 0.0
    intent_classification_ms: float = 0.0
    text_embedding_ms: float = 0.0
    text_search_ms: float = 0.0
    visual_embedding_ms: float = 0.0
    visual_search_ms: float = 0.0
    fusion_ms: float = 0.0
    provenance_audit_ms: float = 0.0
    context_construction_ms: float = 0.0
    llm_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0
    grounding_score: float = 0.0
    hallucination_rate: float = 0.0
    text_candidates_count: int = 0
    visual_candidates_count: int = 0
    fused_results_count: int = 0
    claims_count: int = 0
    supported_claims_count: int = 0
    partially_supported_count: int = 0
    unsupported_claims_count: int = 0
    contradicted_claims_count: int = 0
    insufficient_claims_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_length_chars: int = 0
    response_length_chars: int = 0
    gpu_vram_peak_mb: float = 0.0


class PipelineTracker:
    """Tracks real pipeline execution states and latency metrics."""

    STEP_NAMES = [
        ("Query Processing", "Validating query string and parameters"),
        ("Intent Classification", "Classifying query intent: conversational, knowledge, or investigation"),
        ("BGE Text Embedding", "Generating 384-dim dense representation via BAAI/bge-small-en-v1.5"),
        ("Text FAISS Search", "Searching 1,551-vector text index (IndexFlatIP cosine similarity)"),
        ("CLIP Visual Query Embedding", "Embedding natural language query into 512-dim visual space"),
        ("Visual FAISS Search", "Searching 4-vector visual index for forensic images and diagrams"),
        ("Multimodal Fusion", "Merging cross-modal candidates via RRF or Weighted Score Normalization"),
        ("Epistemic Audit", "Segregating Observed Evidence from Model Interpretations and checking disk files"),
        ("Evidence Context Construction", "Synthesizing structured forensic context with strict epistemic segregation"),
        ("LLM Reasoning", "Grounded cybersecurity reasoning via configured LLM provider"),
        ("Claim Extraction", "Extracting structured claims with supporting evidence IDs and ATT&CK mappings"),
        ("Evidence Verification", "Auditing claim-level support against retrieved forensic evidence"),
        ("Support Classification", "Assigning 5-way verification status and computing grounding metrics"),
        ("Result Presentation", "Assembling investigation answers, cards, and live telemetry"),
    ]

    def __init__(self) -> None:
        self.steps: list[PipelineStep] = [
            PipelineStep(name=name, description=desc) for name, desc in self.STEP_NAMES
        ]
        self.metrics = ExecutionMetrics()
        self._step_start_times: dict[str, float] = {}

    def reset(self) -> None:
        self.steps = [
            PipelineStep(name=name, description=desc) for name, desc in self.STEP_NAMES
        ]
        self.metrics = ExecutionMetrics()
        self._step_start_times.clear()

    def start_step(self, step_name: str, details: str = "") -> None:
        for s in self.steps:
            if s.name == step_name:
                s.status = StepStatus.RUNNING
                if details:
                    s.details = details
                self._step_start_times[step_name] = time.perf_counter()
                break

    def complete_step(self, step_name: str, details: str = "") -> float:
        duration_ms = 0.0
        start_t = self._step_start_times.get(step_name)
        if start_t is not None:
            duration_ms = round((time.perf_counter() - start_t) * 1000, 2)

        for s in self.steps:
            if s.name == step_name:
                s.status = StepStatus.COMPLETE
                s.duration_ms = duration_ms
                if details:
                    s.details = details
                break
        return duration_ms

    def fail_step(self, step_name: str, error_msg: str) -> None:
        for s in self.steps:
            if s.name == step_name:
                s.status = StepStatus.ERROR
                s.details = error_msg
                break

    def skip_step(self, step_name: str, reason: str = "") -> None:
        for s in self.steps:
            if s.name == step_name:
                s.status = StepStatus.SKIPPED
                s.details = reason
                break
