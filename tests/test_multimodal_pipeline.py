"""Unit tests for Phase 3 Multimodal RAG pipeline (OCR, VLM, CLIP, VisualVectorStore, MultimodalRetriever)."""

from pathlib import Path
import numpy as np
import pytest

from src.ingestion.evidence_schema import CanonicalEvidenceRecord
from src.multimodal.ocr_engine import CyberSecurityOCREngine, OCRResult
from src.multimodal.vlm_engine import CyberSecurityVLMEngine, VLMInterpretation
from src.multimodal.visual_embedding_service import CLIPVisualEmbeddingService
from src.multimodal.visual_vector_store import VisualVectorStore
from src.multimodal.image_pipeline import VisualEvidencePipeline
from src.multimodal.multimodal_retriever import MultimodalRetriever
from src.retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"


class TestOCR:
    """Test suite for OCR extraction engine."""

    @pytest.fixture(scope="class")
    @classmethod
    def ocr_engine(cls):
        return CyberSecurityOCREngine()

    def test_ocr_siem_screenshot(self, ocr_engine):
        siem_img = IMAGES_DIR / "screenshots" / "siem_alert_powershell_execution.png"
        assert siem_img.exists()
        res = ocr_engine.extract_text(siem_img)
        assert isinstance(res, OCRResult)
        assert len(res.lines) > 5
        assert "powershell.exe" in res.raw_text.lower()
        assert "executionpolicy" in res.raw_text.lower()
        assert res.confidence > 0.8

    def test_ocr_terminal_screenshot(self, ocr_engine):
        term_img = IMAGES_DIR / "screenshots" / "terminal_credential_access_mimikatz.png"
        assert term_img.exists()
        res = ocr_engine.extract_text(term_img)
        assert "mimikatz" in res.raw_text.lower()
        assert "sekurlsa" in res.raw_text.lower()
        assert "administrator" in res.raw_text.lower()

    def test_ocr_visual_table(self, ocr_engine):
        tbl_img = IMAGES_DIR / "tables" / "visual_threat_intel_ioc_table.png"
        assert tbl_img.exists()
        res = ocr_engine.extract_text(tbl_img)
        assert "indicators of compromise" in res.raw_text.lower()
        assert "t1059.001" in res.raw_text.lower()

    def test_ocr_canonical_format(self, ocr_engine):
        term_img = IMAGES_DIR / "screenshots" / "terminal_credential_access_mimikatz.png"
        res = ocr_engine.extract_text(term_img)
        canonical = res.to_canonical_text("test_image.png")
        assert "[OBSERVED EVIDENCE: OCR TEXT EXTRACTED FROM test_image.png]" in canonical


class TestVLM:
    """Test suite for VLM visual interpretation engine."""

    @pytest.fixture(scope="class")
    @classmethod
    def vlm_engine(cls):
        return CyberSecurityVLMEngine()

    def test_vlm_interpretation_disclaimer(self, vlm_engine):
        diag_img = IMAGES_DIR / "diagrams" / "incident_attack_flow_diagram.png"
        assert diag_img.exists()
        res = vlm_engine.interpret_image(diag_img)
        assert isinstance(res, VLMInterpretation)
        assert res.is_ground_truth is False
        canonical = res.to_canonical_text("diagram.png")
        assert "[MODEL-GENERATED INTERPRETATION - NOT GROUND TRUTH: diagram.png]" in canonical

    def test_vlm_detected_elements(self, vlm_engine):
        siem_img = IMAGES_DIR / "screenshots" / "siem_alert_powershell_execution.png"
        res = vlm_engine.interpret_image(siem_img)
        assert len(res.detected_elements) >= 3
        assert any("Alert" in elem or "PowerShell" in elem for elem in res.detected_elements)


class TestVisualEmbedding:
    """Test suite for CLIP visual embedding service."""

    @pytest.fixture(scope="class")
    @classmethod
    def clip_service(cls):
        return CLIPVisualEmbeddingService()

    def test_clip_embedding_dimension(self, clip_service):
        assert clip_service.dimension == 512

    def test_clip_image_embedding_normalization(self, clip_service):
        siem_img = IMAGES_DIR / "screenshots" / "siem_alert_powershell_execution.png"
        vec = clip_service.embed_image(siem_img)
        assert vec.shape == (1, 512)
        norm = np.linalg.norm(vec[0])
        assert np.isclose(norm, 1.0, atol=1e-3)

    def test_clip_text_query_embedding(self, clip_service):
        query_vec = clip_service.embed_text_query("Mimikatz password dumping")
        assert query_vec.shape == (1, 512)
        norm = np.linalg.norm(query_vec[0])
        assert np.isclose(norm, 1.0, atol=1e-3)


class TestVisualVectorStore:
    """Test suite for VisualVectorStore FAISS index."""

    def test_visual_store_search_and_ranking(self, tmp_path):
        clip_service = CLIPVisualEmbeddingService()
        images = list(IMAGES_DIR.rglob("*.png"))
        assert len(images) >= 4

        embeddings = clip_service.embed_images(images)
        records = [
            {
                "evidence_id": f"IMG-{img.stem}",
                "source_file": img.name,
                "raw_reference": str(img),
                "modality": "image_screenshot",
                "content": f"Image artifact: {img.name}",
                "metadata": {"format": "PNG"},
            }
            for img in images
        ]

        store = VisualVectorStore(dimension=512)
        store.add_records(embeddings, records)
        assert store.total_vectors == len(images)

        # Save and Reload
        store.save(tmp_path)
        reloaded = VisualVectorStore.load(tmp_path, dimension=512)
        assert reloaded.total_vectors == len(images)

        # Search
        q_vec = clip_service.embed_text_query("Mimikatz password dumping in PowerShell terminal")
        results = reloaded.search(q_vec, top_k=2)
        assert len(results) == 2
        assert any("mimikatz" in r.source_file.lower() for r in results)


class TestVisualEvidencePipeline:
    """Test suite for full visual evidence processing pipeline."""

    def test_process_image_creates_canonical_record(self):
        pipeline = VisualEvidencePipeline()
        siem_img = IMAGES_DIR / "screenshots" / "siem_alert_powershell_execution.png"
        rec = pipeline.process_image(siem_img)

        assert isinstance(rec, CanonicalEvidenceRecord)
        assert rec.evidence_id == "IMG-siem_alert_powershell_execution"
        assert rec.modality == "image_screenshot"
        assert rec.source_dataset == "images"
        assert Path(rec.raw_reference).exists()
        assert rec.attck_technique == "T1059.001"
        assert "[OBSERVED EVIDENCE: OCR TEXT" in rec.content
        assert "[MODEL-GENERATED INTERPRETATION - NOT GROUND TRUTH" in rec.content
        assert rec.metadata["ocr_text"] is not None
        assert rec.metadata["vlm_caption"] is not None


class TestMultimodalRetriever:
    """Test suite for joint cross-modal retrieval and fusion."""

    def test_hybrid_retrieval_fusion(self):
        text_index_dir = PROJECT_ROOT / "data" / "processed" / "faiss_index"
        vis_index_dir = PROJECT_ROOT / "data" / "processed" / "visual_faiss_index"

        assert text_index_dir.exists(), "Text FAISS index must exist"
        assert vis_index_dir.exists(), "Visual FAISS index must exist"

        text_store = VectorStore.load(text_index_dir, dimension=384)
        vis_store = VisualVectorStore.load(vis_index_dir, dimension=512)

        retriever = MultimodalRetriever(
            text_store=text_store,
            visual_store=vis_store,
        )

        # Query requiring cross-modal evidence (tested with RRF)
        query = "PowerShell execution policy bypass and Mimikatz credential dumping"
        results = retriever.retrieve(query, top_k=6, fusion_method="rrf")

        assert len(results) == 6
        modalities = [r.modality for r in results]
        # Verify both text and visual modalities are present in candidate ranking
        assert any("image" in m for m in modalities)
        assert any(m in ("mitre_attack", "evtx", "json", "docx", "network_flow", "pdf") for m in modalities)

        # Verify provenance
        for r in results:
            assert Path(r.raw_reference).exists() or (PROJECT_ROOT / r.raw_reference).exists()
            assert r.rank > 0
            assert r.fused_score > 0.0

    def test_weighted_fusion_method(self):
        text_index_dir = PROJECT_ROOT / "data" / "processed" / "faiss_index"
        vis_index_dir = PROJECT_ROOT / "data" / "processed" / "visual_faiss_index"

        text_store = VectorStore.load(text_index_dir, dimension=384)
        vis_store = VisualVectorStore.load(vis_index_dir, dimension=512)

        retriever = MultimodalRetriever(text_store=text_store, visual_store=vis_store)
        results = retriever.retrieve("Log4Shell Apache vulnerability", top_k=3, fusion_method="weighted")
        assert len(results) == 3
        assert results[0].fused_score > 0.0
