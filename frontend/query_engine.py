"""Backend query execution wrapper for the real-time research dashboard.

Coordinates MultimodalRetriever, EvidenceContextBuilder, LLM providers (OpenAI / Mock / Placeholder),
logging, pipeline tracking, and authentic performance metrics.
Phase 3.5: Multimodal RAG + Real LLM Reasoning & Claim Extraction.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings
from src.embeddings.embedding_service import EmbeddingService
from src.llm.base_llm import BaseLLM
from src.llm.context_builder import EvidenceContextBuilder, FormattedEvidenceContext
from src.llm.llm_client import create_llm
from src.llm.mock_provider import MockLLMProvider
from src.llm.openai_provider import OpenAICompatibleProvider
from src.llm.schema import Claim, LLMResponse, StructuredInvestigationAnswer
from src.multimodal.multimodal_retriever import FusedRetrievalResult, MultimodalRetriever
from src.multimodal.ocr_engine import CyberSecurityOCREngine
from src.multimodal.visual_embedding_service import CLIPVisualEmbeddingService
from src.multimodal.visual_vector_store import VisualVectorStore
from src.multimodal.vlm_engine import CyberSecurityVLMEngine
from src.knowledge.cybersecurity_reference_kb import CybersecurityReferenceKnowledgeBase
from src.retrieval.vector_store import RetrievalResult, VectorStore
from src.verification.evidence_verifier import HybridEvidenceVerifier
from src.verification.schema import ClaimVerificationResult, VerificationReport, VerificationStatus
from src.routing.intent_router import QueryIntentRouter, QueryIntent
from frontend.pipeline_tracker import PipelineTracker, StepStatus

logger = logging.getLogger("ResearchDashboard")


@dataclass
class ModelStatusInfo:
    name: str
    model_id: str
    modality: str
    device: str
    dimension: int | str
    is_loaded: bool
    status: str
    vram_usage_mb: float | str = "N/A"


@dataclass
class DatasetStatusInfo:
    name: str
    category: str
    raw_path: str
    record_count: int | str
    status: str


@dataclass
class QueryExecutionResult:
    query: str
    fusion_method: str
    top_k: int
    results: list[FusedRetrievalResult]
    tracker: PipelineTracker
    evidence_context: str = ""
    structured_answer: StructuredInvestigationAnswer | None = None
    llm_response: LLMResponse | None = None
    verification_report: VerificationReport | None = None
    llm_provider_name: str = ""
    error: str | None = None
    is_conversational: bool = False


class DashboardQueryEngine:
    """Manages models, vector indices, context construction, and LLM query execution."""

    def __init__(self) -> None:
        self.settings = Settings()
        self.text_index_dir = PROJECT_ROOT / "data" / "processed" / "faiss_index"
        self.visual_index_dir = PROJECT_ROOT / "data" / "processed" / "visual_faiss_index"

        self.text_store: VectorStore | None = None
        self.visual_store: VisualVectorStore | None = None
        self.text_embed_service: EmbeddingService | None = None
        self.visual_embed_service: CLIPVisualEmbeddingService | None = None
        self.vlm_engine: CyberSecurityVLMEngine | None = None
        self.ocr_engine: CyberSecurityOCREngine | None = None
        self.retriever: MultimodalRetriever | None = None

        self.context_builder: EvidenceContextBuilder | None = None
        self.verifier: HybridEvidenceVerifier | None = None
        self.llm_provider: BaseLLM | None = None
        self.llm_provider_name: str = "mock"

        self._initialized = False

    def initialize(self, llm_provider_override: str | None = None) -> None:
        """Lazy load or initialize stores, services, context builder, and LLM."""
        if self._initialized and llm_provider_override is None:
            return

        logger.info("Initializing DashboardQueryEngine (Phase 4)...")

        # 1. Load Text Store
        if self.text_index_dir.exists() and self.text_store is None:
            try:
                self.text_store = VectorStore.load(
                    self.text_index_dir,
                    dimension=self.settings.embedding_dimension,
                )
                logger.info("Text VectorStore loaded: %d vectors", self.text_store._index.ntotal)
            except Exception as e:
                logger.error("Failed to load text index: %s", e)

        # 2. Load Visual Store
        if self.visual_index_dir.exists() and self.visual_store is None:
            try:
                self.visual_store = VisualVectorStore.load(
                    self.visual_index_dir,
                    dimension=512,
                )
                logger.info("Visual VectorStore loaded: %d vectors", self.visual_store._index.ntotal)
            except Exception as e:
                logger.error("Failed to load visual index: %s", e)

        # 3. Initialize Embedding Services & OCR
        if self.text_embed_service is None:
            self.text_embed_service = EmbeddingService(model_name=self.settings.embedding_model_name)
        if self.visual_embed_service is None:
            self.visual_embed_service = CLIPVisualEmbeddingService()
        if self.ocr_engine is None:
            self.ocr_engine = CyberSecurityOCREngine()

        # 4. Multimodal Retriever
        if self.text_store is not None and self.retriever is None:
            self.retriever = MultimodalRetriever(
                text_store=self.text_store,
                visual_store=self.visual_store,
                text_embedding_service=self.text_embed_service,
                visual_embedding_service=self.visual_embed_service,
                text_weight=0.65,
                visual_weight=0.35,
            )

        # 5. Evidence Context Builder
        if self.context_builder is None:
            self.context_builder = EvidenceContextBuilder()

        # 6. Hybrid Evidence Verifier (Phase 4)
        if self.verifier is None:
            self.verifier = HybridEvidenceVerifier()

        # 7. LLM Provider Resolution
        # If explicitly passed or if OPENAI_API_KEY is present in env, select appropriate provider
        target_provider = llm_provider_override or self.settings.llm_provider
        if target_provider in ["placeholder", "openai"] and not os.environ.get("OPENAI_API_KEY"):
            if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("LLM_PROVIDER") == "gemini":
                target_provider = "gemini"
            elif os.environ.get("GROQ_API_KEY"):
                target_provider = "groq"
            else:
                target_provider = "mock"
        elif target_provider == "placeholder":
            target_provider = "mock"

        self.set_llm_provider(target_provider)
        self._initialized = True
        logger.info("DashboardQueryEngine initialized (Active LLM Provider: %s)", self.llm_provider_name)

    def set_llm_provider(self, provider_name: str, **kwargs: Any) -> None:
        """Switch active LLM provider dynamically."""
        provider_clean = provider_name.lower().strip()
        try:
            params = {
                "model_name": self.settings.llm_model_name or None,
                "api_key": self.settings.llm_api_key or None,
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_tokens,
            }
            params.update(kwargs)
            self.llm_provider = create_llm(
                provider=provider_clean,
                **params,
            )
            self.llm_provider_name = provider_clean
            logger.info("LLM provider switched to: %s (%s)", provider_clean, self.llm_provider.__class__.__name__)
        except Exception as exc:
            logger.warning("Failed to switch LLM provider to '%s': %s. Reverting to MockLLMProvider.", provider_clean, exc)
            self.llm_provider = MockLLMProvider()
            self.llm_provider_name = "mock"

    def get_model_statuses(self) -> list[ModelStatusInfo]:
        """Inspect actual runtime status of loaded models."""
        self.initialize()
        models: list[ModelStatusInfo] = []

        # 1. BGE Text Embedder
        bge_loaded = bool(self.text_embed_service and self.text_embed_service._model is not None)
        bge_device = "NOT LOADED"
        bge_dim = self.settings.embedding_dimension
        if bge_loaded:
            try:
                bge_device = str(self.text_embed_service._model.device)
            except Exception:
                bge_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        models.append(
            ModelStatusInfo(
                name="BGE-small Text Embedder",
                model_id=self.settings.embedding_model_name,
                modality="Dense Text (384-dim)",
                device=bge_device,
                dimension=bge_dim,
                is_loaded=bge_loaded,
                status="READY" if bge_loaded else "STANDBY (Loads on query)",
            )
        )

        # 2. CLIP Visual Embedder
        clip_loaded = bool(self.visual_embed_service and self.visual_embed_service._model is not None)
        clip_device = "NOT LOADED"
        if clip_loaded:
            try:
                clip_device = str(next(self.visual_embed_service._model.parameters()).device)
            except Exception:
                clip_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        models.append(
            ModelStatusInfo(
                name="OpenAI CLIP ViT-B/32",
                model_id="openai/clip-vit-base-patch32",
                modality="Cross-Modal Vision & Query (512-dim)",
                device=clip_device,
                dimension=512,
                is_loaded=clip_loaded,
                status="READY" if clip_loaded else "STANDBY (Loads on query)",
            )
        )

        # 3. Salesforce BLIP VLM
        blip_loaded = bool(self.vlm_engine and self.vlm_engine._model is not None)
        blip_device = "NOT LOADED"
        if blip_loaded:
            try:
                blip_device = str(next(self.vlm_engine._model.parameters()).device)
            except Exception:
                blip_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        models.append(
            ModelStatusInfo(
                name="Salesforce BLIP VLM",
                model_id="Salesforce/blip-image-captioning-base",
                modality="Vision-Language Reasoning",
                device=blip_device,
                dimension="~224M params",
                is_loaded=blip_loaded,
                status="READY" if blip_loaded else "STANDBY",
            )
        )

        # 4. OCR Engine
        models.append(
            ModelStatusInfo(
                name="CyberSecurity OCR Engine",
                model_id="cyber_ocr_siem_profile",
                modality="Optical Text Extraction",
                device="CPU",
                dimension="Text Lines & Tokens",
                is_loaded=True,
                status="READY",
            )
        )

        # 5. LLM Reasoning Engine
        llm_name = self.llm_provider.provider_name if self.llm_provider else "Not Initialized"
        llm_model = getattr(self.llm_provider, "model_name", "N/A")
        llm_dev = "API (Cloud / Network)" if "OpenAI" in llm_name else "In-Memory (Local / Mock)"
        models.append(
            ModelStatusInfo(
                name=f"LLM: {llm_name}",
                model_id=llm_model,
                modality="Grounded Reasoning & JSON Claims",
                device=llm_dev,
                dimension="Structured JSON Schema",
                is_loaded=bool(self.llm_provider),
                status="READY" if self.llm_provider else "OFFLINE",
            )
        )

        return models

    def get_dataset_statuses(self) -> list[DatasetStatusInfo]:
        """Inspect actual indexes and raw data stores."""
        self.initialize()
        datasets: list[DatasetStatusInfo] = []

        # Text FAISS
        t_count = self.text_store._index.ntotal if self.text_store else 0
        datasets.append(
            DatasetStatusInfo(
                name="Unified Text FAISS Store",
                category="Vector Index",
                raw_path=str(self.text_index_dir),
                record_count=f"{t_count:,} vectors",
                status="READY" if t_count > 0 else "MISSING",
            )
        )

        # Visual FAISS
        v_count = self.visual_store._index.ntotal if self.visual_store else 0
        datasets.append(
            DatasetStatusInfo(
                name="Visual FAISS Store (CLIP)",
                category="Vector Index",
                raw_path=str(self.visual_index_dir),
                record_count=f"{v_count:,} vectors",
                status="READY" if v_count > 0 else "MISSING",
            )
        )

        # Raw datasets inspection
        raw_items = [
            ("MITRE ATT&CK Enterprise", "Knowledge Base", "data/raw/mitre_attack/enterprise-attack.json"),
            ("NVD CVE Vulnerabilities", "Vulnerability Feed", "data/raw/nvd/nvd_cve_sample.json"),
            ("Atomic-EVTX Windows Logs", "Host Telemetry", "data/raw/atomic_evtx"),
            ("OTRF Mordor Adversary Logs", "Adversary Simulation", "data/raw/mordor"),
            ("CIC-IDS2017 Flows", "Network Telemetry", "data/raw/cic_ids2017/cicids2017_representative_sample.csv"),
            ("Forensic Reports (PDF/DOCX)", "Unstructured Documents", "data/sample"),
            ("Visual Evidence Screenshots", "Forensic Visual Artifacts", "data/raw/images"),
        ]

        for name, category, rel_path in raw_items:
            full_p = PROJECT_ROOT / rel_path
            count_str = "Available"
            status = "FOUND" if full_p.exists() else "MISSING"

            if full_p.exists() and full_p.is_dir():
                file_count = len(list(full_p.glob("*.*")))
                count_str = f"{file_count} files"
            elif full_p.exists() and full_p.is_file():
                size_kb = full_p.stat().st_size / 1024
                count_str = f"{size_kb:.1f} KB"

            datasets.append(
                DatasetStatusInfo(
                    name=name,
                    category=category,
                    raw_path=rel_path,
                    record_count=count_str,
                    status=status,
                )
            )

        return datasets

    def execute_query(
        self,
        query: str,
        top_k: int = 5,
        fusion_method: str = "weighted",
        llm_provider_override: str | None = None,
        session_evidence: list[FusedRetrievalResult] | None = None,
        **kwargs: Any,
    ) -> QueryExecutionResult:
        """Execute the full end-to-end Phase 3.5 Multimodal RAG + LLM Reasoning pipeline."""

        tracker = PipelineTracker()
        total_start = time.perf_counter()

        logger.info("==================================================")
        logger.info("Query received: '%s'", query)

        # Step 1: Query Processing
        tracker.start_step("Query Processing", f"Validating query length and sanitizing: '{query[:40]}...'")
        q_clean = query.strip()
        if not q_clean:
            tracker.fail_step("Query Processing", "Empty query string provided")
            return QueryExecutionResult(
                query=query,
                fusion_method=fusion_method,
                top_k=top_k,
                results=[],
                tracker=tracker,
                error="Query string cannot be empty.",
            )
        tracker.complete_step("Query Processing", f"{len(q_clean)} characters")

        # Step 1.5: Intent Classification (pre-retrieval gate, <1ms, no GPU)
        tracker.start_step("Intent Classification", "Classifying query intent")
        intent_start = time.perf_counter()
        router = QueryIntentRouter()
        intent_result = router.classify(q_clean, has_file_attachment=bool(session_evidence))
        intent_ms = round((time.perf_counter() - intent_start) * 1000, 2)
        tracker.metrics.intent_classification_ms = intent_ms
        tracker.complete_step(
            "Intent Classification",
            f"{intent_result.intent.value} ({intent_result.confidence:.0%}) in {intent_ms:.1f}ms — {intent_result.reason}",
        )
        logger.info(
            "Intent classified: %s (confidence=%.2f, reason=%s) in %.1fms",
            intent_result.intent.value, intent_result.confidence, intent_result.reason, intent_ms,
        )

        # Short-circuit: conversational queries bypass the entire investigation pipeline
        if intent_result.intent == QueryIntent.CONVERSATIONAL:
            logger.info("Conversational intent — skipping retrieval, LLM investigation, and verification.")
            conversational_text = router.get_conversational_response(q_clean)

            # Skip all downstream investigation steps
            skip_reason = "Conversational query — no investigation needed"
            for step_name in [
                "BGE Text Embedding", "Text FAISS Search",
                "CLIP Visual Query Embedding", "Visual FAISS Search",
                "Multimodal Fusion", "Epistemic Audit",
                "Evidence Context Construction", "LLM Reasoning",
                "Claim Extraction", "Evidence Verification",
                "Support Classification",
            ]:
                tracker.skip_step(step_name, skip_reason)

            tracker.start_step("Result Presentation", "Assembling conversational response")
            tracker.complete_step("Result Presentation", "Conversational response — zero retrieval")

            total_ms = round((time.perf_counter() - total_start) * 1000, 2)
            tracker.metrics.total_latency_ms = total_ms

            return QueryExecutionResult(
                query=q_clean,
                fusion_method=fusion_method,
                top_k=top_k,
                results=[],
                tracker=tracker,
                structured_answer=StructuredInvestigationAnswer(
                    answer=conversational_text,
                    claims=[],
                    confidence="high",
                    limitations=[],
                ),
                is_conversational=True,
            )

        # Initialize models only for non-conversational queries
        self.initialize(llm_provider_override=llm_provider_override)

        # Step 2: Dense Text Embedding
        logger.info("BGE text embedding started")
        tracker.start_step("BGE Text Embedding", "Encoding with BAAI/bge-small-en-v1.5")
        t_embed_start = time.perf_counter()
        query_vec = self.text_embed_service.embed_query(q_clean)
        t_embed_ms = round((time.perf_counter() - t_embed_start) * 1000, 2)
        device_used = "cuda:0" if torch.cuda.is_available() else "cpu"
        tracker.metrics.text_embedding_ms = t_embed_ms
        tracker.complete_step("BGE Text Embedding", f"{t_embed_ms:.1f}ms on {device_used}")
        logger.info("BGE embedding generated in %.1fms (device: %s)", t_embed_ms, device_used)

        # Step 3: Text FAISS Search
        logger.info("Text retrieval started")
        tracker.start_step("Text FAISS Search", f"Searching {self.text_store._index.ntotal} vectors")
        t_search_start = time.perf_counter()
        raw_candidates = self.text_store.search(query_vec, top_k=max(top_k * 6, 30))

        # Inject authoritative Reference Knowledge Base records (semantic + keyword match)
        ref_kb_results = CybersecurityReferenceKnowledgeBase.search_reference_knowledge(
            q_clean, query_embedding=query_vec, top_k=2, min_score=0.55,
        )
        if ref_kb_results:
            logger.info("Injected %d reference knowledge records for concept query", len(ref_kb_results))

        combined_candidates = ref_kb_results + raw_candidates
        text_candidates = MultimodalRetriever._rerank_with_query_keywords(q_clean, combined_candidates)
        text_candidates = text_candidates[:max(top_k * 2, 10)]

        t_search_ms = round((time.perf_counter() - t_search_start) * 1000, 2)
        tracker.metrics.text_search_ms = t_search_ms
        tracker.metrics.text_candidates_count = len(text_candidates)
        tracker.complete_step("Text FAISS Search", f"Retrieved & reranked {len(text_candidates)} candidates in {t_search_ms:.1f}ms")
        logger.info("Text FAISS search and reranking returned %d candidates in %.1fms", len(text_candidates), t_search_ms)

        # Step 4: CLIP Visual Query Embedding
        logger.info("CLIP visual query embedding started")
        visual_candidates = []
        if self.visual_store is not None:
            tracker.start_step("CLIP Visual Query Embedding", "Mapping query to 512-dim visual space")
            v_embed_start = time.perf_counter()
            vis_query_vec = self.visual_embed_service.embed_text_query(q_clean)
            v_embed_ms = round((time.perf_counter() - v_embed_start) * 1000, 2)
            tracker.metrics.visual_embedding_ms = v_embed_ms
            clip_dev = self.visual_embed_service.device
            tracker.complete_step("CLIP Visual Query Embedding", f"{v_embed_ms:.1f}ms on {clip_dev}")
            logger.info("CLIP visual query embedded in %.1fms on %s", v_embed_ms, clip_dev)

            # Step 5: Visual FAISS Search
            logger.info("Visual retrieval started")
            tracker.start_step("Visual FAISS Search", f"Searching {self.visual_store._index.ntotal} visual vectors")
            v_search_start = time.perf_counter()
            visual_candidates = self.visual_store.search(vis_query_vec, top_k=max(2, top_k))
            v_search_ms = round((time.perf_counter() - v_search_start) * 1000, 2)
            tracker.metrics.visual_search_ms = v_search_ms
            tracker.metrics.visual_candidates_count = len(visual_candidates)
            tracker.complete_step("Visual FAISS Search", f"Retrieved {len(visual_candidates)} images in {v_search_ms:.1f}ms")
            logger.info("Visual FAISS search returned %d candidates in %.1fms", len(visual_candidates), v_search_ms)
        else:
            tracker.skip_step("CLIP Visual Query Embedding", "Visual store not available")
            tracker.skip_step("Visual FAISS Search", "Visual store not available")

        # Step 6: Multimodal Fusion
        logger.info("Multimodal fusion started (%s)", fusion_method)
        tracker.start_step("Multimodal Fusion", f"Fusing candidates via {fusion_method}")
        fusion_start = time.perf_counter()
        if fusion_method == "rrf":
            fused = self.retriever._fuse_rrf(text_candidates, visual_candidates)
        else:
            fused = self.retriever._fuse_weighted(text_candidates, visual_candidates, min_visual_threshold=0.2)
        fused_results = fused[:top_k]
        if session_evidence:
            fused_results = list(session_evidence) + [r for r in fused_results if r.evidence_id not in {s.evidence_id for s in session_evidence}][:max(1, top_k)]
        for r_idx, r_item in enumerate(fused_results, start=1):
            r_item.rank = r_idx
        fusion_ms = round((time.perf_counter() - fusion_start) * 1000, 2)
        tracker.metrics.fusion_ms = fusion_ms
        tracker.metrics.fused_results_count = len(fused_results)
        tracker.complete_step("Multimodal Fusion", f"Ranked {len(fused_results)} candidates in {fusion_ms:.1f}ms")
        logger.info("Multimodal fusion completed in %.1fms", fusion_ms)

        # Step 7: Epistemic Audit & Provenance Verification
        tracker.start_step("Epistemic Audit", "Verifying disk provenance and separating observed vs reference evidence")
        audit_start = time.perf_counter()
        for res in fused_results:
            is_user_up = (
                str(res.source_dataset).upper() == "USER_UPLOAD"
                or res.metadata.get("is_user_provided", False)
                or res.evidence_id.startswith("UPLOAD-")
            )
            if is_user_up:
                res.metadata["provenance_type"] = "USER_UPLOAD"
                res.source_dataset = "USER_UPLOAD"
                if "MODEL-GENERATED" in res.content or res.metadata.get("evidence_nature") == "MODEL_INTERPRETATION":
                    res.evidence_nature = "MODEL_INTERPRETATION"
                else:
                    res.evidence_nature = "OBSERVED_EVIDENCE"
            else:
                res.metadata["provenance_type"] = "KNOWLEDGE_BASE"
                if not res.source_dataset:
                    res.source_dataset = "KNOWLEDGE_BASE"
                res.evidence_nature = MultimodalRetriever._classify_evidence_nature(res.content, res.metadata)

            res.metadata["evidence_nature"] = res.evidence_nature

            raw_p = Path(res.raw_reference)
            if not raw_p.exists():
                raw_p = PROJECT_ROOT / res.raw_reference
            res.metadata["file_exists_on_disk"] = raw_p.exists()
            res.metadata["disk_resolved_path"] = str(raw_p)

        audit_ms = round((time.perf_counter() - audit_start) * 1000, 2)
        tracker.metrics.provenance_audit_ms = audit_ms
        tracker.complete_step("Epistemic Audit", f"100% provenance checked ({len(fused_results)} items) in {audit_ms:.1f}ms")
        logger.info("Epistemic audit completed in %.1fms", audit_ms)

        # Step 8: Evidence Context Construction (Phase 3.5)
        tracker.start_step("Evidence Context Construction", "Formatting top-K evidence with epistemic segregation")
        ctx_start = time.perf_counter()
        formatted_ctx: FormattedEvidenceContext = self.context_builder.build_context(
            query=q_clean,
            results=fused_results,
        )
        ctx_ms = round((time.perf_counter() - ctx_start) * 1000, 2)
        tracker.metrics.context_construction_ms = ctx_ms
        tracker.metrics.prompt_length_chars = len(formatted_ctx.context_text)
        tracker.complete_step("Evidence Context Construction", f"{len(formatted_ctx.context_text)} chars in {ctx_ms:.1f}ms")
        logger.info("Evidence context built (%d chars, %d items) in %.1fms", len(formatted_ctx.context_text), len(fused_results), ctx_ms)

        # Step 9: LLM Reasoning & Generation (Phase 3.5)
        active_llm = self.llm_provider or MockLLMProvider()
        active_provider_name = getattr(active_llm, "provider_name", "LLMProvider")
        active_model_name = getattr(active_llm, "model_name", "unknown")

        tracker.start_step("LLM Reasoning", f"Request started to {active_provider_name} ({active_model_name})")
        logger.info("LLM request started (%s, model=%s)", active_provider_name, active_model_name)

        llm_response: LLMResponse = active_llm.generate_response(
            prompt=q_clean,
            context=formatted_ctx.context_text,
        )

        tracker.metrics.llm_latency_ms = llm_response.latency_ms
        tracker.metrics.prompt_tokens = llm_response.prompt_tokens
        tracker.metrics.completion_tokens = llm_response.completion_tokens
        tracker.metrics.total_tokens = llm_response.total_tokens
        tracker.metrics.response_length_chars = len(llm_response.text)

        pipeline_error = None
        if llm_response.error:
            logger.warning("LLM generation encountered issue: %s", llm_response.error)
            tracker.fail_step("LLM Reasoning", llm_response.error)
            pipeline_error = llm_response.error
        else:
            tracker.complete_step("LLM Reasoning", f"{llm_response.latency_ms:.1f}ms, {llm_response.total_tokens} tokens")
            logger.info("LLM response received in %.1fms (%d tokens)", llm_response.latency_ms, llm_response.total_tokens)

        # Step 10: Claim Extraction (Phase 3.5)
        tracker.start_step("Claim Extraction", "Extracting structured claims, citations, and ATT&CK/CVE mappings")
        structured_answer = llm_response.structured
        if structured_answer and structured_answer.claims:
            claims_count = len(structured_answer.claims)
            tracker.metrics.claims_count = claims_count
            tracker.complete_step("Claim Extraction", f"{claims_count} claims extracted with citations")
            logger.info("Claim extraction completed (%d claims identified)", claims_count)
        else:
            if pipeline_error:
                tracker.skip_step("Claim Extraction", "LLM failed — no claims extracted")
            else:
                tracker.complete_step("Claim Extraction", "0 discrete claims parsed (general summary)")
                logger.info("Claim extraction completed (0 claims)")

        # Step 11: Evidence Verification (Phase 4)
        tracker.start_step("Evidence Verification", "Auditing claim-level support against retrieved forensic evidence")
        logger.info("Phase 4 claim verification started")
        verif_start = time.perf_counter()

        verification_report = None
        if structured_answer and structured_answer.claims:
            verifier = self.verifier or HybridEvidenceVerifier()
            verification_report = verifier.verify_all(
                claims=structured_answer.claims,
                evidence=fused_results,
                query=q_clean,
                raw_answer=structured_answer.answer,
            )
            verif_ms = round((time.perf_counter() - verif_start) * 1000, 2)
            tracker.metrics.verification_latency_ms = verif_ms
            tracker.metrics.grounding_score = verification_report.overall_grounding_score
            tracker.metrics.hallucination_rate = verification_report.hallucination_rate
            tracker.metrics.supported_claims_count = verification_report.supported_count
            tracker.metrics.partially_supported_count = verification_report.partially_supported_count
            tracker.metrics.unsupported_claims_count = verification_report.unsupported_count
            tracker.metrics.contradicted_claims_count = verification_report.contradicted_count
            tracker.metrics.insufficient_claims_count = verification_report.insufficient_evidence_count
            tracker.complete_step("Evidence Verification", f"Audited {len(structured_answer.claims)} claims in {verif_ms:.1f}ms")
            logger.info(
                "Evidence verification completed: %d supported, %d partial, %d unsupported, %d contradicted, %d insufficient (Grounding: %.2f)",
                verification_report.supported_count,
                verification_report.partially_supported_count,
                verification_report.unsupported_count,
                verification_report.contradicted_count,
                verification_report.insufficient_evidence_count,
                verification_report.overall_grounding_score,
            )

            # Step 12: Support Classification
            tracker.start_step("Support Classification", "Assigning 5-way verification status and computing grounding metrics")
            classif_summary = (
                f"Grounding: {verification_report.overall_grounding_score * 100:.1f}% | "
                f"Hallucination: {verification_report.hallucination_rate * 100:.1f}%"
            )
            tracker.complete_step("Support Classification", classif_summary)
        else:
            if pipeline_error:
                tracker.skip_step("Evidence Verification", "No claims to verify due to LLM error")
                tracker.skip_step("Support Classification", "Skipped")
            else:
                tracker.complete_step("Evidence Verification", "No discrete claims to audit")
                tracker.complete_step("Support Classification", "N/A (no claims)")

        # Step 13: Result Presentation
        tracker.start_step("Result Presentation", "Assembling investigation cards, findings, and telemetry")
        tracker.complete_step("Result Presentation", f"Assembled {len(fused_results)} evidence units and findings")
        logger.info("Result displayed to dashboard")

        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        tracker.metrics.total_latency_ms = total_ms

        if torch.cuda.is_available():
            tracker.metrics.gpu_vram_peak_mb = round(torch.cuda.max_memory_allocated(0) / (1024**2), 1)

        logger.info("Investigation query completed successfully in %.1fms", total_ms)

        return QueryExecutionResult(
            query=q_clean,
            fusion_method=fusion_method,
            top_k=top_k,
            results=fused_results,
            tracker=tracker,
            evidence_context=formatted_ctx.context_text,
            structured_answer=structured_answer,
            llm_response=llm_response,
            verification_report=verification_report,
            llm_provider_name=active_provider_name,
            error=pipeline_error,
        )
