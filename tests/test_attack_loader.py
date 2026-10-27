"""Unit tests for MITRE ATT&CK STIX knowledge loader and indexer."""

from pathlib import Path
import pytest

from src.knowledge.attack_loader import AttackKnowledgeBase

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="class")
def attack_kb(tmp_path_factory):
    stix_file = PROJECT_ROOT / "data" / "raw" / "mitre_attack" / "enterprise-attack.json"
    assert stix_file.exists(), "enterprise-attack.json must be present in data/raw/mitre_attack"

    tmp_dir = tmp_path_factory.mktemp("attack_test")
    index_file = tmp_dir / "attack_index.json"

    kb = AttackKnowledgeBase()
    kb.build_from_stix(stix_file, index_file)
    return kb, index_file


class TestAttackKnowledgeBase:
    """Test suite for ATT&CK STIX ingestion and lookups."""

    def test_build_and_load_index(self, attack_kb):
        kb, index_file = attack_kb
        assert kb.index_loaded
        assert len(kb.techniques) > 500

        # Test reload from index file
        reloaded_kb = AttackKnowledgeBase(index_path=index_file)
        assert reloaded_kb.index_loaded
        assert len(reloaded_kb.techniques) == len(kb.techniques)

    def test_lookup_powershell_technique(self, attack_kb):
        kb, _ = attack_kb
        tech = kb.get_technique("T1059.001")
        assert tech is not None
        assert tech["technique_id"] == "T1059.001"
        assert "PowerShell" in tech["name"]
        assert "execution" in tech["tactics"]

    def test_lookup_lsass_dumping_technique(self, attack_kb):
        kb, _ = attack_kb
        tech = kb.get_technique("T1003.001")
        assert tech is not None
        assert "LSASS" in tech["name"] or "Memory" in tech["name"]
        assert "credential-access" in tech["tactics"]

    def test_technique_search(self, attack_kb):
        kb, _ = attack_kb
        results = kb.search_techniques("kerberos", limit=3)
        assert len(results) > 0
        names_and_descs = [r["name"].lower() + " " + r["description"].lower() for r in results]
        assert any("kerberos" in nd for nd in names_and_descs)

    def test_to_canonical_records(self, attack_kb):
        kb, _ = attack_kb
        records = kb.to_canonical_records()
        assert len(records) == len(kb.techniques)
        sample = records[0]
        assert sample.modality == "mitre_attack"
        assert sample.event_type == "knowledge_technique"
        assert sample.evidence_id.startswith("ATTCK-")
        assert len(sample.content) > 50
