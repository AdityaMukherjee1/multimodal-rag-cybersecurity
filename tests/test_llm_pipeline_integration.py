"""End-to-end integration tests for Multimodal RAG + LLM Reasoning.

Validates the full pipeline flow:
Question -> Multimodal Retrieval -> Epistemic Context Building -> LLM Reasoning -> Structured Answer.

Automated tests execute with MockLLMProvider (zero network, deterministic).
A separate optional live test is guarded by RUN_LIVE_LLM_TESTS=1 and OPENAI_API_KEY.
"""

import os
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.embeddings.embedding_service import EmbeddingService
from src.llm.context_builder import EvidenceContextBuilder
from src.llm.mock_provider import MockLLMProvider
from src.llm.openai_provider import OpenAICompatibleProvider
from src.multimodal.multimodal_retriever import MultimodalRetriever
from src.multimodal.visual_embedding_service import CLIPVisualEmbeddingService
from src.multimodal.visual_vector_store import VisualVectorStore
from src.retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestMultimodalLLMIntegration:
    @classmethod
    def setup_class(cls):
        """Load actual text and visual FAISS stores for realistic integration."""
        cls.settings = Settings()
        text_index_dir = PROJECT_ROOT / "data" / "processed" / "faiss_index"
        visual_index_dir = PROJECT_ROOT / "data" / "processed" / "visual_faiss_index"

        assert text_index_dir.exists(), "Text index directory missing"
        assert visual_index_dir.exists(), "Visual index directory missing"

        cls.text_store = VectorStore.load(text_index_dir, dimension=384)
        cls.visual_store = VisualVectorStore.load(visual_index_dir, dimension=512)
        cls.text_embed = EmbeddingService(cls.settings.embedding_model_name)
        cls.visual_embed = CLIPVisualEmbeddingService()

        cls.retriever = MultimodalRetriever(
            text_store=cls.text_store,
            visual_store=cls.visual_store,
            text_embedding_service=cls.text_embed,
            visual_embedding_service=cls.visual_embed,
        )
        cls.context_builder = EvidenceContextBuilder()
        cls.mock_llm = MockLLMProvider()

    def test_end_to_end_mock_pipeline_credential_dumping(self):
        query = "What evidence indicates credential dumping?"

        # 1. Retrieval
        fused_results = self.retriever.retrieve(query=query, top_k=5, fusion_method="weighted")
        assert len(fused_results) > 0

        # 2. Context Building
        ctx = self.context_builder.build_context(query=query, results=fused_results)
        assert ctx.total_items == len(fused_results)
        assert len(ctx.evidence_ids) == len(fused_results)
        assert "[OBSERVED EVIDENCE:" in ctx.context_text

        # 3. LLM Reasoning
        response = self.mock_llm.generate_response(
            prompt=query,
            context=ctx.context_text,
        )

        # 4. Verification of Structured Output & Grounding
        assert response.is_success
        assert response.structured is not None
        assert response.structured.is_parsed_json
        assert len(response.structured.claims) > 0

        # Verify claims reference the actual retrieved evidence IDs
        all_cited_ids = set()
        for claim in response.structured.claims:
            assert len(claim.supporting_evidence_ids) > 0
            all_cited_ids.update(claim.supporting_evidence_ids)

        # At least one cited ID should be in the retrieved evidence IDs
        intersection = all_cited_ids.intersection(set(ctx.evidence_ids))
        assert len(intersection) > 0, f"Cited {all_cited_ids} does not intersect retrieved {ctx.evidence_ids}"

        # Token and latency metrics
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.total_tokens == response.prompt_tokens + response.completion_tokens
        assert response.latency_ms >= 0

    def test_end_to_end_mock_pipeline_log4shell(self):
        query = "What is CVE-2021-44228?"
        fused_results = self.retriever.retrieve(query=query, top_k=3, fusion_method="rrf")
        ctx = self.context_builder.build_context(query=query, results=fused_results)
        response = self.mock_llm.generate_response(prompt=query, context=ctx.context_text)

        assert response.is_success
        assert "CVE-2021-44228" in response.structured.cves
        assert len(response.structured.claims) > 0


class TestLiveLLMIntegrationOptional:
    """Explicit live test. Only runs if OPENAI_API_KEY and RUN_LIVE_LLM_TESTS=1 are set."""

    @pytest.mark.skipif(
        not (os.environ.get("OPENAI_API_KEY") and os.environ.get("RUN_LIVE_LLM_TESTS") == "1"),
        reason="Skipped: requires OPENAI_API_KEY in environment and RUN_LIVE_LLM_TESTS=1",
    )
    def test_live_openai_api_call(self):
        provider = OpenAICompatibleProvider()
        assert provider.is_configured

        prompt = "What evidence indicates PowerShell execution policy bypass?"
        sample_context = (
            "[DOCX-001]\n"
            "Modality: docx\n"
            "Content: Observed process creation powershell.exe -ExecutionPolicy Bypass -File script.ps1.\n"
            "ATT&CK Technique: T1059.001\n"
        )

        response = provider.generate_response(prompt=prompt, context=sample_context)
        assert response.is_success, f"Live call failed: {response.error}"
        assert response.structured is not None
        assert len(response.structured.claims) > 0
        assert response.total_tokens > 0
