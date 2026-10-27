"""Ablation Study Framework for Phase 5 Empirical Analysis.

Systematically measures the contribution of each architectural component:
1. Full Proposed System
2. Without Visual Retrieval (Text RAG + Verification)
3. Without Multimodal Fusion (Naive Concatenation without Score Normalization)
4. Without Claim Verification (Unverified Multimodal Claims)
5. Without ATT&CK / CVE Knowledge Integration (Forensic Telemetry Only)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from src.evaluation.configurations import EvaluationPipelineManager, ExperimentRunOutput
from src.multimodal.multimodal_retriever import FusedRetrievalResult, MultimodalRetriever

logger = logging.getLogger(__name__)


class AblationStudyRunner:
    """Executes ablation variations on identical evaluation benchmark queries."""

    def __init__(self, manager: EvaluationPipelineManager) -> None:
        self.manager = manager
        self.retriever = manager.retriever
        self.llm = manager.llm
        self.context_builder = manager.context_builder
        self.verifier = manager.verifier

    def run_ablation_no_visual(
        self,
        query: str,
        query_id: str,
        category: str,
        top_k: int = 5,
    ) -> ExperimentRunOutput:
        """Ablation 1: W/O Visual Retrieval (Text RAG + Verification)."""
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
                    source_file=m.get("source_file", "unknown"),
                    source_dataset=m.get("source_dataset", "general"),
                    modality=m.get("modality", "text"),
                    content=tr.content,
                    raw_reference=m.get("raw_reference", ""),
                    metadata=m,
                    fused_score=tr.score,
                    rank=idx,
                    evidence_nature=MultimodalRetriever._classify_evidence_nature(tr.content, m),
                )
            )

        ctx = self.context_builder.build_context(query=query, results=fused_items)
        llm_res = self.llm.generate_response(prompt=query, context=ctx.context_text)

        claims = []
        cited_ids = []
        claim_objs = []
        if llm_res.structured and llm_res.structured.claims:
            claim_objs = llm_res.structured.claims
            for c in claim_objs:
                claims.append(c.to_dict())
                cited_ids.extend(c.supporting_evidence_ids or [])

        t_verif_start = time.perf_counter()
        report = self.verifier.verify_all(
            claims=claim_objs,
            evidence=fused_items,
            query=query,
            raw_answer=llm_res.structured.answer if llm_res.structured else llm_res.text,
        )
        verif_ms = round((time.perf_counter() - t_verif_start) * 1000, 2)
        total_ms = round((time.perf_counter() - start_total) * 1000, 2)
        gpu_mb = round(torch.cuda.max_memory_allocated(0) / (1024**2), 1) if torch.cuda.is_available() else 0.0

        return ExperimentRunOutput(
            config_name="Ablation_WO_Visual",
            query_id=query_id,
            query=query,
            category=category,
            retrieved_evidence_ids=ret_ids,
            answer_text=report.verified_answer,
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

    def run_ablation_no_fusion(
        self,
        query: str,
        query_id: str,
        category: str,
        top_k: int = 5,
    ) -> ExperimentRunOutput:
        """Ablation 2: W/O Multimodal Fusion.

        Retrieves candidates independently and naively appends without score normalization or RRF.
        """
        start_total = time.perf_counter()

        t_ret_start = time.perf_counter()
        q_emb = self.retriever.text_embed.embed_query(query)
        text_results = self.retriever.text_store.search(q_emb, top_k=top_k)

        vis_results = []
        if self.retriever.visual_store is not None:
            vis_q = self.retriever.visual_embed.embed_text_query(query)
            vis_results = self.retriever.visual_store.search(vis_q, top_k=top_k)

        ret_ms = round((time.perf_counter() - t_ret_start) * 1000, 2)

        naive_fused: list[FusedRetrievalResult] = []
        idx = 1
        for tr in text_results[: max(1, top_k // 2)]:
            m = tr.metadata or {}
            eid = m.get("evidence_id") or f"TEXT-{idx:03d}"
            naive_fused.append(
                FusedRetrievalResult(
                    evidence_id=eid,
                    source_file=m.get("source_file", "unknown"),
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
            idx += 1

        for vr in vis_results[: max(1, top_k - len(naive_fused))]:
            naive_fused.append(
                FusedRetrievalResult(
                    evidence_id=vr.evidence_id,
                    source_file=vr.source_file,
                    source_dataset=vr.metadata.get("source_dataset", "images"),
                    modality=vr.modality,
                    content=vr.content,
                    raw_reference=vr.raw_reference,
                    metadata=vr.metadata,
                    fused_score=vr.score,
                    rank=idx,
                    evidence_nature="OBSERVED_EVIDENCE" if vr.metadata.get("ocr_text") else "MODEL_INTERPRETATION",
                )
            )
            idx += 1

        ret_ids = [item.evidence_id for item in naive_fused]
        ctx = self.context_builder.build_context(query=query, results=naive_fused)
        llm_res = self.llm.generate_response(prompt=query, context=ctx.context_text)

        claims = []
        cited_ids = []
        claim_objs = []
        if llm_res.structured and llm_res.structured.claims:
            claim_objs = llm_res.structured.claims
            for c in claim_objs:
                claims.append(c.to_dict())
                cited_ids.extend(c.supporting_evidence_ids or [])

        t_verif_start = time.perf_counter()
        report = self.verifier.verify_all(
            claims=claim_objs,
            evidence=naive_fused,
            query=query,
            raw_answer=llm_res.structured.answer if llm_res.structured else llm_res.text,
        )
        verif_ms = round((time.perf_counter() - t_verif_start) * 1000, 2)
        total_ms = round((time.perf_counter() - start_total) * 1000, 2)
        gpu_mb = round(torch.cuda.max_memory_allocated(0) / (1024**2), 1) if torch.cuda.is_available() else 0.0

        return ExperimentRunOutput(
            config_name="Ablation_WO_Fusion",
            query_id=query_id,
            query=query,
            category=category,
            retrieved_evidence_ids=ret_ids,
            answer_text=report.verified_answer,
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

    def run_ablation_no_verification(
        self,
        query: str,
        query_id: str,
        category: str,
        top_k: int = 5,
    ) -> ExperimentRunOutput:
        """Ablation 3: W/O Claim Verification (Raw Multimodal RAG with unverified claims)."""
        res = self.manager.run_config_c_multimodal_rag(
            query=query,
            query_id=query_id,
            category=category,
            top_k=top_k,
        )
        res.config_name = "Ablation_WO_Verification"
        return res

    def run_ablation_no_knowledge_base(
        self,
        query: str,
        query_id: str,
        category: str,
        top_k: int = 5,
    ) -> ExperimentRunOutput:
        """Ablation 4: W/O ATT&CK / CVE Knowledge Integration."""
        start_total = time.perf_counter()

        t_ret_start = time.perf_counter()
        q_emb = self.retriever.text_embed.embed_query(query)
        raw_text_results = self.retriever.text_store.search(q_emb, top_k=top_k * 4)

        filtered_text: list[Any] = []
        for tr in raw_text_results:
            src = (tr.metadata.get("source_file") or "").lower()
            mod = (tr.metadata.get("modality") or "").lower()
            if "enterprise-attack" in src or "nvd_cve" in src or "mitre" in mod or "cve" in mod:
                continue
            filtered_text.append(tr)
            if len(filtered_text) >= top_k:
                break

        vis_results = []
        if self.retriever.visual_store is not None:
            vis_q = self.retriever.visual_embed.embed_text_query(query)
            vis_results = self.retriever.visual_store.search(vis_q, top_k=max(2, top_k))

        fused = self.retriever._fuse_weighted(filtered_text, vis_results, min_visual_threshold=0.2)[:top_k]
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
            config_name="Ablation_WO_KnowledgeBase",
            query_id=query_id,
            query=query,
            category=category,
            retrieved_evidence_ids=ret_ids,
            answer_text=report.verified_answer,
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
