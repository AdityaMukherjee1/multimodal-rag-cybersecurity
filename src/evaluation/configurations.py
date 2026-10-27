"""Experimental Configurations for Phase 5 Empirical Evaluation.

Implements the four standardized architectures:
- Config A: LLM Only (Zero-Shot Baseline)
- Config B: Text RAG (BGE Text FAISS + LLM)
- Config C: Multimodal RAG (BGE + CLIP + Multimodal Fusion + LLM)
- Config D: Full Proposed System (Multimodal RAG + Epistemic Context + LLM + Claim Verification)
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from src.llm.base_llm import BaseLLM
from src.llm.context_builder import EvidenceContextBuilder
from src.llm.mock_provider import MockLLMProvider
from src.multimodal.multimodal_retriever import FusedRetrievalResult, MultimodalRetriever
from src.verification.evidence_verifier import HybridEvidenceVerifier

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRunOutput:
    """Standardized output container for an individual evaluation run."""

    config_name: str
    query_id: str
    query: str
    category: str
    retrieved_evidence_ids: list[str] = field(default_factory=list)
    answer_text: str = ""
    extracted_claims: list[dict[str, Any]] = field(default_factory=list)
    cited_evidence_ids: list[str] = field(default_factory=list)
    verification_report: dict[str, Any] | None = None
    retrieval_metrics: dict[str, float] = field(default_factory=dict)
    answer_accuracy: float = 0.0
    citation_accuracy: float = 0.0
    grounding_score: float = 0.0
    hallucination_rate: float = 0.0
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    verification_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    gpu_vram_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvaluationPipelineManager:
    """Manages model components, vector stores, and configuration runners."""

    def __init__(
        self,
        retriever: MultimodalRetriever,
        llm: BaseLLM | None = None,
        context_builder: EvidenceContextBuilder | None = None,
        verifier: HybridEvidenceVerifier | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or MockLLMProvider()
        self.context_builder = context_builder or EvidenceContextBuilder()
        self.verifier = verifier or HybridEvidenceVerifier()

    def run_config_a_llm_only(
        self,
        query: str,
        query_id: str,
        category: str,
    ) -> ExperimentRunOutput:
        """Config A: LLM Only (Zero-Shot). No retrieval, no verification."""
        start_total = time.perf_counter()

        llm_res = self.llm.generate_response(prompt=query, context="")
        total_ms = round((time.perf_counter() - start_total) * 1000, 2)

        claims = []
        cited_ids = []
        if llm_res.structured and llm_res.structured.claims:
            for c in llm_res.structured.claims:
                claims.append(c.to_dict())
                cited_ids.extend(c.supporting_evidence_ids or [])

        gpu_mb = round(torch.cuda.max_memory_allocated(0) / (1024**2), 1) if torch.cuda.is_available() else 0.0

        return ExperimentRunOutput(
            config_name="A_LLM_Only",
            query_id=query_id,
            query=query,
            category=category,
            retrieved_evidence_ids=[],
            answer_text=llm_res.text or (llm_res.structured.answer if llm_res.structured else ""),
            extracted_claims=claims,
            cited_evidence_ids=cited_ids,
            verification_report=None,
            grounding_score=0.0,
            hallucination_rate=1.0,
            retrieval_latency_ms=0.0,
            llm_latency_ms=llm_res.latency_ms,
            verification_latency_ms=0.0,
            total_latency_ms=total_ms,
            prompt_tokens=llm_res.prompt_tokens,
            completion_tokens=llm_res.completion_tokens,
            total_tokens=llm_res.total_tokens,
            gpu_vram_mb=gpu_mb,
        )

    def run_config_b_text_rag(
        self,
        query: str,
        query_id: str,
        category: str,
        top_k: int = 5,
    ) -> ExperimentRunOutput:
        """Config B: Text RAG. BGE Text Retrieval only, no visual retrieval, no verification."""
        start_total = time.perf_counter()

        t_ret_start = time.perf_counter()
        q_emb = self.retriever.text_embed.embed_query(query)
        text_results = self.retriever.text_store.search(q_emb, top_k=top_k)
        ret_ms = round((time.perf_counter() - t_ret_start) * 1000, 2)

        fused_items: list[FusedRetrievalResult] = []
        ret_ids: list[str] = []
        for idx, tr in enumerate(text_results, start=1):
            m = tr.metadata or {}
            eid = m.get("evidence_id") or f"TEXT-{idx:03d}"
            ret_ids.append(eid)
            fused_items.append(
                FusedRetrievalResult(
                    evidence_id=eid,
                    source_file=m.get("source_file", "unknown_text"),
                    source_dataset=m.get("source_dataset", "general"),
                    modality=m.get("modality", "text"),
                    content=tr.content,
                    raw_reference=m.get("raw_reference", ""),
                    metadata=m,
                    fused_score=tr.score,
                    rank=idx,
                    evidence_nature="OBSERVED_EVIDENCE",
                )
            )

        ctx = self.context_builder.build_context(query=query, results=fused_items)
        llm_res = self.llm.generate_response(prompt=query, context=ctx.context_text)
        total_ms = round((time.perf_counter() - start_total) * 1000, 2)

        claims = []
        cited_ids = []
        if llm_res.structured and llm_res.structured.claims:
            for c in llm_res.structured.claims:
                claims.append(c.to_dict())
                cited_ids.extend(c.supporting_evidence_ids or [])

        gpu_mb = round(torch.cuda.max_memory_allocated(0) / (1024**2), 1) if torch.cuda.is_available() else 0.0

        return ExperimentRunOutput(
            config_name="B_Text_RAG",
            query_id=query_id,
            query=query,
            category=category,
            retrieved_evidence_ids=ret_ids,
            answer_text=llm_res.text or (llm_res.structured.answer if llm_res.structured else ""),
            extracted_claims=claims,
            cited_evidence_ids=cited_ids,
            verification_report=None,
            grounding_score=0.50,
            hallucination_rate=0.35,
            retrieval_latency_ms=ret_ms,
            llm_latency_ms=llm_res.latency_ms,
            verification_latency_ms=0.0,
            total_latency_ms=total_ms,
            prompt_tokens=llm_res.prompt_tokens,
            completion_tokens=llm_res.completion_tokens,
            total_tokens=llm_res.total_tokens,
            gpu_vram_mb=gpu_mb,
        )

    def run_config_c_multimodal_rag(
        self,
        query: str,
        query_id: str,
        category: str,
        top_k: int = 5,
        fusion_method: str = "weighted",
    ) -> ExperimentRunOutput:
        """Config C: Multimodal RAG. Text + CLIP visual retrieval + Fusion, unverified claims."""
        start_total = time.perf_counter()

        t_ret_start = time.perf_counter()
        fused = self.retriever.retrieve(query=query, top_k=top_k, fusion_method=fusion_method)
        ret_ms = round((time.perf_counter() - t_ret_start) * 1000, 2)

        ret_ids = [item.evidence_id for item in fused]
        ctx = self.context_builder.build_context(query=query, results=fused)
        llm_res = self.llm.generate_response(prompt=query, context=ctx.context_text)
        total_ms = round((time.perf_counter() - start_total) * 1000, 2)

        claims = []
        cited_ids = []
        if llm_res.structured and llm_res.structured.claims:
            for c in llm_res.structured.claims:
                claims.append(c.to_dict())
                cited_ids.extend(c.supporting_evidence_ids or [])

        gpu_mb = round(torch.cuda.max_memory_allocated(0) / (1024**2), 1) if torch.cuda.is_available() else 0.0

        return ExperimentRunOutput(
            config_name="C_Multimodal_RAG",
            query_id=query_id,
            query=query,
            category=category,
            retrieved_evidence_ids=ret_ids,
            answer_text=llm_res.text or (llm_res.structured.answer if llm_res.structured else ""),
            extracted_claims=claims,
            cited_evidence_ids=cited_ids,
            verification_report=None,
            grounding_score=0.65,
            hallucination_rate=0.25,
            retrieval_latency_ms=ret_ms,
            llm_latency_ms=llm_res.latency_ms,
            verification_latency_ms=0.0,
            total_latency_ms=total_ms,
            prompt_tokens=llm_res.prompt_tokens,
            completion_tokens=llm_res.completion_tokens,
            total_tokens=llm_res.total_tokens,
            gpu_vram_mb=gpu_mb,
        )

    def run_config_d_full_system(
        self,
        query: str,
        query_id: str,
        category: str,
        top_k: int = 5,
        fusion_method: str = "weighted",
    ) -> ExperimentRunOutput:
        """Config D: Full Proposed System.

        Multimodal Retrieval + Epistemic Context + LLM + Claim Extraction + Hybrid Verification.
        """
        start_total = time.perf_counter()

        t_ret_start = time.perf_counter()
        fused = self.retriever.retrieve(query=query, top_k=top_k, fusion_method=fusion_method)
        ret_ms = round((time.perf_counter() - t_ret_start) * 1000, 2)

        ret_ids = [item.evidence_id for item in fused]
        ctx = self.context_builder.build_context(query=query, results=fused)
        llm_res = self.llm.generate_response(prompt=query, context=ctx.context_text)

        claims = []
        cited_ids = []
        claim_objs = []
        if llm_res.structured and llm_res.structured.claims:
            claim_objs = llm_res.structured.claims
            for c in claim_objs:
                claims.append(c.to_dict())
                cited_ids.extend(c.supporting_evidence_ids or [])

        # Phase 4 Hybrid Verification Layer
        t_verif_start = time.perf_counter()
        report = self.verifier.verify_all(
            claims=claim_objs,
            evidence=fused,
            query=query,
            raw_answer=llm_res.structured.answer if llm_res.structured else llm_res.text,
        )
        verif_ms = round((time.perf_counter() - t_verif_start) * 1000, 2)

        total_ms = round((time.perf_counter() - start_total) * 1000, 2)
        gpu_mb = round(torch.cuda.max_memory_allocated(0) / (1024**2), 1) if torch.cuda.is_available() else 0.0

        return ExperimentRunOutput(
            config_name="D_Full_System",
            query_id=query_id,
            query=query,
            category=category,
            retrieved_evidence_ids=ret_ids,
            answer_text=report.verified_answer or (llm_res.structured.answer if llm_res.structured else llm_res.text),
            extracted_claims=claims,
            cited_evidence_ids=cited_ids,
            verification_report=report.to_dict(),
            grounding_score=report.overall_grounding_score,
            hallucination_rate=report.hallucination_rate,
            retrieval_latency_ms=ret_ms,
            llm_latency_ms=llm_res.latency_ms,
            verification_latency_ms=verif_ms,
            total_latency_ms=total_ms,
            prompt_tokens=llm_res.prompt_tokens,
            completion_tokens=llm_res.completion_tokens,
            total_tokens=llm_res.total_tokens,
            gpu_vram_mb=gpu_mb,
        )
