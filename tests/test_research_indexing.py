"""Integration tests for Phase 2 research data indexing and multi-modal retrieval."""

from pathlib import Path
import pytest

from src.embeddings.embedding_service import EmbeddingService
from src.retrieval.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestResearchIndexing:
    """Test suite for research data FAISS index and retrieval."""

    @pytest.fixture(scope="class")
    @classmethod
    def research_store(cls):
        index_dir = PROJECT_ROOT / "data" / "processed" / "faiss_index"
        assert index_dir.exists(), "FAISS index directory must exist"
        store = VectorStore.load(index_dir, dimension=384)
        return store

    @pytest.fixture(scope="class")
    @classmethod
    def embed_service(cls):
        return EmbeddingService()

    def test_index_has_expected_vectors(self, research_store):
        assert research_store._index.ntotal >= 1000
        assert len(research_store._documents) == research_store._index.ntotal

    def test_multi_modal_provenance_present(self, research_store):
        docs = research_store._documents
        modalities = set(d.get("metadata", {}).get("modality") for d in docs)
        assert "mitre_attack" in modalities
        assert "nvd_cve" in modalities
        assert "evtx" in modalities
        assert "json" in modalities
        assert "network_flow" in modalities
        assert "pdf" in modalities
        assert "docx" in modalities

    def test_retrieve_powershell_attack_technique(self, research_store, embed_service):
        query = "PowerShell command execution technique"
        query_vec = embed_service.embed_texts([query])
        results = research_store.search(query_vec, top_k=5)

        assert len(results) == 5
        # Verify at least one top result is related to PowerShell or T1059
        contents = [r.content for r in results]
        assert any("T1059" in c or "PowerShell" in c or "powershell" in c for c in contents)
        assert all(r.score > 0.0 for r in results)

    def test_retrieve_log4shell_cve(self, research_store, embed_service):
        query = "Log4Shell CVE-2021-44228 JNDI injection vulnerability"
        query_vec = embed_service.embed_texts([query])
        results = research_store.search(query_vec, top_k=5)

        assert len(results) == 5
        contents = [r.content for r in results]
        assert any("CVE-2021-44228" in c or "Log4j" in c for c in contents)

    def test_retrieve_evtx_dcsync_event(self, research_store, embed_service):
        query = "Directory service replication DCSync event ID 4662"
        query_vec = embed_service.embed_texts([query])
        results = research_store.search(query_vec, top_k=5)

        assert len(results) == 5
        contents = [r.content for r in results]
        assert any("4662" in c or "DCSync" in c or "T1003" in c for c in contents)
