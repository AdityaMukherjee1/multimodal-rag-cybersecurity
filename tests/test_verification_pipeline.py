"""End-to-end integration test for Phase 4 Claim Verification Pipeline.

Validates the complete chain:
Question -> Multimodal Retrieval -> Epistemic Context -> LLM Reasoning -> Claim Extraction -> Evidence Verification -> Verified Report.
"""

from pathlib import Path

from src.config.settings import Settings
from src.embeddings.embedding_service import EmbeddingService
from src.llm.context_builder import EvidenceContextBuilder
from src.llm.mock_provider import MockLLMProvider
from src.multimodal.multimodal_retriever import MultimodalRetriever
from src.multimodal.visual_embedding_service import CLIPVisualEmbeddingService
from src.multimodal.visual_vector_store import VisualVectorStore
from src.retrieval.vector_store import VectorStore
from src.verification.evidence_verifier import HybridEvidenceVerifier
from src.verification.schema import VerificationStatus

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestVerificationPipelineIntegration:
    @classmethod
    def setup_class(cls):
        cls.settings = Settings()
        text_index_dir = PROJECT_ROOT / "data" / "processed" / "faiss_index"
        visual_index_dir = PROJECT_ROOT / "data" / "processed" / "visual_faiss_index"

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
        cls.verifier = HybridEvidenceVerifier()

    def test_full_retrieval_reasoning_verification_pipeline(self):
        query = "What evidence indicates credential dumping?"

        # 1. Multimodal Retrieval & Fusion
        fused = self.retriever.retrieve(query=query, top_k=5, fusion_method="weighted")
        assert len(fused) > 0

        # 2. Context Building
        ctx = self.context_builder.build_context(query=query, results=fused)

        # 3. LLM Reasoning & Claim Generation
        llm_res = self.mock_llm.generate_response(prompt=query, context=ctx.context_text)
        assert llm_res.is_success
        assert llm_res.structured is not None
        claims = llm_res.structured.claims
        assert len(claims) > 0

        # 4. Phase 4 Evidence Verification
        report = self.verifier.verify_all(
            claims=claims,
            evidence=fused,
            query=query,
            raw_answer=llm_res.structured.answer,
        )

        assert report.total_claims == len(claims)
        assert report.supported_count > 0
        assert report.overall_grounding_score >= 0.70
        assert report.hallucination_rate == 0.0
        assert len(report.verified_answer) > 0

        # Verify individual claim result structure
        c1 = report.claim_results[0]
        assert c1.status in (VerificationStatus.SUPPORTED, VerificationStatus.PARTIALLY_SUPPORTED)
        assert len(c1.provenance_sources) > 0
