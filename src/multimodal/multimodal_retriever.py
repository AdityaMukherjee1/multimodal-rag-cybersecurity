"""Multimodal retriever and evidence fusion engine.

Coordinates joint multi-source retrieval across:
1. Text & Structured Knowledge Store (BGE-small 384-dim FAISS index)
2. Visual Evidence Store (CLIP 512-dim Visual FAISS index)

Implements Reciprocal Rank Fusion (RRF) and Weighted Score Normalization
to rank heterogeneous cybersecurity evidence (EVTX, Mordor, MITRE ATT&CK,
NVD CVEs, Network flows, PDFs, DOCX tables, and visual screenshots/diagrams)
while maintaining complete provenance and evidence classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.embeddings.embedding_service import EmbeddingService
from src.multimodal.visual_embedding_service import CLIPVisualEmbeddingService
from src.multimodal.visual_vector_store import VisualRetrievalResult, VisualVectorStore
from src.retrieval.vector_store import RetrievalResult, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class FusedRetrievalResult:
    """A single fused evidence retrieval result across text and visual modalities.

    Attributes:
        evidence_id: Canonical evidence identifier (e.g. 'EVTX-001', 'IMG-siem_alert').
        source_file: Original filename of the evidence.
        source_dataset: Originating dataset (e.g. 'atomic_evtx', 'images', 'mitre_attack').
        modality: Modality tag (evtx, json, network_flow, pdf, docx, image_screenshot, etc.).
        content: Canonical text summary or visual OCR/interpretation.
        raw_reference: File path pointing directly to the original artifact on disk.
        metadata: Domain-specific forensic attributes.
        fused_score: Combined ranking score (higher = more relevant).
        text_score: Similarity score from text embedding search (if matched).
        visual_score: Similarity score from visual CLIP search (if matched).
        retrieval_modality: 'text', 'visual', or 'hybrid_fused'.
        evidence_nature: 'OBSERVED_EVIDENCE' or 'MODEL_INTERPRETATION'.
        rank: Final rank position (1-based).
    """

    evidence_id: str
    source_file: str
    source_dataset: str
    modality: str
    content: str
    raw_reference: str
    metadata: dict[str, Any] = field(default_factory=dict)
    fused_score: float = 0.0
    text_score: float | None = None
    visual_score: float | None = None
    retrieval_modality: str = "text"
    evidence_nature: str = "OBSERVED_EVIDENCE"
    rank: int = 0


class MultimodalRetriever:
    """Orchestrates cross-modal retrieval across text and visual cybersecurity vector stores."""

    def __init__(
        self,
        text_store: VectorStore,
        visual_store: VisualVectorStore | None = None,
        text_embedding_service: EmbeddingService | None = None,
        visual_embedding_service: CLIPVisualEmbeddingService | None = None,
        text_weight: float = 0.65,
        visual_weight: float = 0.35,
        rrf_k: int = 60,
    ) -> None:
        """Initialize the multimodal retriever.

        Args:
            text_store: VectorStore instance for text/knowledge embeddings.
            visual_store: Optional VisualVectorStore for image embeddings.
            text_embedding_service: Service for generating text embeddings (BGE).
            visual_embedding_service: Service for generating visual embeddings (CLIP).
            text_weight: Weight assigned to text similarity scores (0.0 to 1.0).
            visual_weight: Weight assigned to visual similarity scores (0.0 to 1.0).
            rrf_k: Smoothing constant for Reciprocal Rank Fusion.
        """
        self.text_store = text_store
        self.visual_store = visual_store
        self.text_embed = text_embedding_service or EmbeddingService()
        self.visual_embed = visual_embedding_service or CLIPVisualEmbeddingService()
        self.text_weight = text_weight
        self.visual_weight = visual_weight
        self.rrf_k = rrf_k
        logger.info(
            "MultimodalRetriever initialized (text_weight=%.2f, visual_weight=%.2f, rrf_k=%d)",
            text_weight, visual_weight, rrf_k,
        )

    @staticmethod
    def _classify_evidence_nature(content: str, metadata: dict[str, Any]) -> str:
        """Determine the epistemic nature of an evidence record.

        Returns:
            'REFERENCE_KNOWLEDGE' for knowledge base entries (ATT&CK, CVE/NVD, PDFs, DOCX, CWE, reference KB).
            'MODEL_INTERPRETATION' for AI-generated content (VLM captions, BLIP descriptions).
            'OBSERVED_EVIDENCE' for raw telemetry (EVTX events, Sysmon logs, network flows, user uploads).
        """
        # Explicit metadata override (e.g. from reference KB injection)
        if metadata.get("evidence_nature") == "REFERENCE_KNOWLEDGE":
            return "REFERENCE_KNOWLEDGE"
        if metadata.get("is_reference_knowledge"):
            return "REFERENCE_KNOWLEDGE"

        # AI-generated content
        if "MODEL-GENERATED" in content:
            return "MODEL_INTERPRETATION"

        # Knowledge base datasets (not raw telemetry)
        dataset = metadata.get("source_dataset", "").lower()
        modality = metadata.get("modality", "").lower()
        kb_datasets = {
            "mitre_attack", "nvd", "documents_pdf", "documents_docx",
            "reference_knowledge", "vulnerability_kb",
        }
        kb_modalities = {
            "mitre_attack", "nvd_cve", "pdf", "docx",
            "vulnerability_knowledge", "reference_handbook",
        }
        if dataset in kb_datasets or modality in kb_modalities:
            return "REFERENCE_KNOWLEDGE"

        return "OBSERVED_EVIDENCE"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "rrf",
        min_visual_threshold: float = 0.15,
    ) -> list[FusedRetrievalResult]:
        """Perform multimodal retrieval and evidence fusion for an investigation query.

        Args:
            query: Natural language investigator query.
            top_k: Total number of fused evidence items to return.
            fusion_method: 'rrf' (reciprocal rank fusion, default) or 'weighted' (score-weighted).
            min_visual_threshold: Minimum cosine score to include visual candidate.

        Returns:
            List of ranked FusedRetrievalResult objects.
        """
        logger.info("Executing multimodal retrieval for query: '%s' (top_k=%d, fusion=%s)", query, top_k, fusion_method)

        # 1. Text Search (BGE 384-dim)
        text_vec = self.text_embed.embed_texts([query])
        text_results: list[RetrievalResult] = self.text_store.search(text_vec, top_k=max(top_k * 6, 30))
        text_results = self._rerank_with_query_keywords(query, text_results)

        # 2. Visual Search (CLIP 512-dim)
        visual_results: list[VisualRetrievalResult] = []
        if self.visual_store and self.visual_store.total_vectors > 0:
            try:
                vis_query_vec = self.visual_embed.embed_text_query(query)
                visual_results = self.visual_store.search(vis_query_vec, top_k=top_k)
            except Exception as e:
                logger.error("Visual retrieval error: %s", e)

        # 3. Fuse Candidate Pools
        if fusion_method == "rrf":
            fused = self._fuse_rrf(text_results, visual_results)
        else:
            fused = self._fuse_weighted(text_results, visual_results, min_visual_threshold)

        # Truncate to top_k and assign final ranks
        final_results = fused[:top_k]
        for rank, res in enumerate(final_results, start=1):
            res.rank = rank

        logger.info(
            "Multimodal retrieval completed: returned %d fused results (%d text, %d visual candidates inspected)",
            len(final_results), len(text_results), len(visual_results),
        )
        return final_results

    @staticmethod
    def _rerank_with_query_keywords(
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Apply query-aware keyword alignment reranking to dense retrieval candidates.

        Boosts candidates containing salient query terms (e.g. 'sql' in 'What is SQL injection?')
        and penalizes candidates missing distinctive keywords when other candidates have them.
        This fixes semantic drift where BGE dense retrieval matches lexically similar but
        semantically unrelated techniques (e.g. T1055 Process Injection vs SQL Injection).
        """
        import re as _re

        stop_words = {
            "what", "is", "a", "an", "the", "and", "or", "in", "on", "of", "to",
            "for", "with", "this", "that", "these", "those", "how", "explain",
            "describe", "does", "do", "can", "about", "tell", "me", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "it", "its",
            "my", "your", "we", "they", "you", "i",
        }
        q_tokens = [
            w for w in _re.findall(r"\b\w+\b", query.lower())
            if w not in stop_words and len(w) > 1
        ]
        if not q_tokens:
            return candidates

        scored: list[tuple[float, RetrievalResult]] = []
        for r in candidates:
            combined_text = (r.content + " " + str(r.metadata)).lower()
            token_matches = sum(1 for tok in q_tokens if tok in combined_text)
            overlap_ratio = token_matches / len(q_tokens)

            hybrid_score = r.score * (1.0 + 0.35 * overlap_ratio)

            # Penalize candidates missing distinctive query terms
            if "sql" in q_tokens and "sql" not in combined_text:
                hybrid_score *= 0.65
            if "phish" in query.lower() and "phish" not in combined_text:
                hybrid_score *= 0.70
            if "cve-2021-44228" in query.lower() and "cve-2021-44228" in combined_text:
                hybrid_score *= 1.4

            scored.append((hybrid_score, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for new_rank, (score, r) in enumerate(scored, start=1):
            r.score = score
            r.rank = new_rank
            reranked.append(r)
        return reranked

    def _fuse_weighted(
        self,
        text_results: list[RetrievalResult],
        visual_results: list[VisualRetrievalResult],
        min_visual_threshold: float,
    ) -> list[FusedRetrievalResult]:
        """Combine candidates using normalized weighted similarity scores."""
        candidates: dict[str, FusedRetrievalResult] = {}

        # Normalize text scores to [0.5, 1.0] to preserve relative rank while aligning ranges
        t_scores = [r.score for r in text_results]
        if t_scores:
            min_t, max_t = min(t_scores), max(t_scores)
            range_t = (max_t - min_t) if (max_t - min_t) > 1e-6 else 1.0
            norm_t = [(s - min_t) / range_t for s in t_scores]
        else:
            norm_t = []

        # Process text results
        for idx, res in enumerate(text_results):
            meta = res.metadata
            ev_id = meta.get("evidence_id", f"TXT-{res.rank}")
            src_file = meta.get("source_file", meta.get("filename", "unknown"))
            dataset = meta.get("source_dataset", "text_corpus")
            modality = meta.get("modality", "text")
            raw_ref = meta.get("raw_reference", "")

            nature = self._classify_evidence_nature(res.content, meta)

            normalized_score = norm_t[idx] if idx < len(norm_t) else res.score
            fused_score = (normalized_score * 0.5 + 0.5) * self.text_weight

            candidates[ev_id] = FusedRetrievalResult(
                evidence_id=ev_id,
                source_file=src_file,
                source_dataset=dataset,
                modality=modality,
                content=res.content,
                raw_reference=raw_ref,
                metadata=meta,
                fused_score=fused_score,
                text_score=res.score,
                visual_score=None,
                retrieval_modality="text",
                evidence_nature=nature,
            )

        # Normalize visual scores to [0.5, 1.0] to match text scale
        v_scores = [r.score for r in visual_results if r.score >= min_visual_threshold]
        if v_scores:
            min_v, max_v = min(v_scores), max(v_scores)
            range_v = (max_v - min_v) if (max_v - min_v) > 1e-6 else 1.0
            norm_v = {r.evidence_id: (r.score - min_v) / range_v for r in visual_results if r.score >= min_visual_threshold}
        else:
            norm_v = {}

        # Process visual results
        for res in visual_results:
            if res.score < min_visual_threshold:
                continue

            ev_id = res.evidence_id
            nature = self._classify_evidence_nature(res.content, res.metadata)

            normalized_score = norm_v.get(ev_id, 0.5)
            fused_score = (normalized_score * 0.5 + 0.5) * self.visual_weight

            if ev_id in candidates:
                existing = candidates[ev_id]
                existing.visual_score = res.score
                existing.fused_score = existing.fused_score + fused_score
                existing.retrieval_modality = "hybrid_fused"
            else:
                candidates[ev_id] = FusedRetrievalResult(
                    evidence_id=ev_id,
                    source_file=res.source_file,
                    source_dataset="images",
                    modality=res.modality,
                    content=res.content,
                    raw_reference=res.raw_reference,
                    metadata=res.metadata,
                    fused_score=fused_score,
                    text_score=None,
                    visual_score=res.score,
                    retrieval_modality="visual",
                    evidence_nature=nature,
                )

        # Sort descending by fused score
        return sorted(candidates.values(), key=lambda x: x.fused_score, reverse=True)

    def _fuse_rrf(
        self,
        text_results: list[RetrievalResult],
        visual_results: list[VisualRetrievalResult],
    ) -> list[FusedRetrievalResult]:
        """Combine candidates using Reciprocal Rank Fusion (RRF)."""
        candidates: dict[str, FusedRetrievalResult] = {}
        rrf_scores: dict[str, float] = {}

        # Text RRF
        for rank, res in enumerate(text_results, start=1):
            meta = res.metadata
            ev_id = meta.get("evidence_id", f"TXT-{rank}")
            src_file = meta.get("source_file", meta.get("filename", "unknown"))
            dataset = meta.get("source_dataset", "text_corpus")
            modality = meta.get("modality", "text")
            raw_ref = meta.get("raw_reference", "")

            nature = self._classify_evidence_nature(res.content, meta)

            rrf_score = 1.0 / (self.rrf_k + rank)
            rrf_scores[ev_id] = rrf_scores.get(ev_id, 0.0) + rrf_score

            candidates[ev_id] = FusedRetrievalResult(
                evidence_id=ev_id,
                source_file=src_file,
                source_dataset=dataset,
                modality=modality,
                content=res.content,
                raw_reference=raw_ref,
                metadata=meta,
                fused_score=rrf_score,
                text_score=res.score,
                visual_score=None,
                retrieval_modality="text",
                evidence_nature=nature,
            )

        # Visual RRF
        for rank, res in enumerate(visual_results, start=1):
            ev_id = res.evidence_id
            rrf_score = 1.0 / (self.rrf_k + rank)
            rrf_scores[ev_id] = rrf_scores.get(ev_id, 0.0) + rrf_score

            nature = self._classify_evidence_nature(res.content, res.metadata)

            if ev_id in candidates:
                existing = candidates[ev_id]
                existing.visual_score = res.score
                existing.fused_score = rrf_scores[ev_id]
                existing.retrieval_modality = "hybrid_fused"
            else:
                candidates[ev_id] = FusedRetrievalResult(
                    evidence_id=ev_id,
                    source_file=res.source_file,
                    source_dataset="images",
                    modality=res.modality,
                    content=res.content,
                    raw_reference=res.raw_reference,
                    metadata=res.metadata,
                    fused_score=rrf_score,
                    text_score=None,
                    visual_score=res.score,
                    retrieval_modality="visual",
                    evidence_nature=nature,
                )

        for ev_id, c in candidates.items():
            c.fused_score = rrf_scores.get(ev_id, 0.0)

        return sorted(candidates.values(), key=lambda x: x.fused_score, reverse=True)
