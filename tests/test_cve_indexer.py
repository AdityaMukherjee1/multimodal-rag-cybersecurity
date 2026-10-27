"""Unit tests for NVD CVE vulnerability indexer."""

from pathlib import Path
import pytest

from src.knowledge.cve_indexer import CVEKnowledgeBase

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="class")
def cve_kb(tmp_path_factory):
    nvd_file = PROJECT_ROOT / "data" / "raw" / "nvd" / "nvd_cve_sample.json"
    assert nvd_file.exists(), "nvd_cve_sample.json must be present in data/raw/nvd"

    tmp_dir = tmp_path_factory.mktemp("cve_test")
    index_file = tmp_dir / "cve_index.json"

    kb = CVEKnowledgeBase()
    kb.build_from_nvd(nvd_file, index_file)
    return kb, index_file


class TestCVEKnowledgeBase:
    """Test suite for NVD CVE ingestion and lookups."""

    def test_build_and_load_cve_index(self, cve_kb):
        kb, index_file = cve_kb
        assert kb.index_loaded
        assert len(kb.cves) >= 7

        reloaded_kb = CVEKnowledgeBase(index_path=index_file)
        assert reloaded_kb.index_loaded
        assert len(reloaded_kb.cves) == len(kb.cves)

    def test_lookup_log4shell(self, cve_kb):
        kb, _ = cve_kb
        cve = kb.get_cve("CVE-2021-44228")
        assert cve is not None
        assert cve["cve_id"] == "CVE-2021-44228"
        assert "Log4j" in cve["description"]
        assert cve["cvss"]["base_score"] == 10.0
        assert cve["cvss"]["severity"] == "CRITICAL"

    def test_lookup_eternalblue(self, cve_kb):
        kb, _ = cve_kb
        cve = kb.get_cve("CVE-2017-0144")
        assert cve is not None
        assert "SMBv1" in cve["description"]

    def test_search_cves(self, cve_kb):
        kb, _ = cve_kb
        results = kb.search_cves("Exchange", limit=2)
        assert len(results) >= 1
        assert any(r["cve_id"] == "CVE-2021-26855" for r in results)

    def test_to_canonical_cve_records(self, cve_kb):
        kb, _ = cve_kb
        records = kb.to_canonical_records()
        assert len(records) == len(kb.cves)
        sample = records[0]
        assert sample.modality == "nvd_cve"
        assert sample.event_type == "vulnerability_record"
        assert sample.evidence_id.startswith("CVE-")
        assert sample.cve_id is not None
