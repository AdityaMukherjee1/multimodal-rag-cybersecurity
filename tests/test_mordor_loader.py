"""Unit tests for Mordor structured NDJSON loader."""

from pathlib import Path
import pytest

from src.ingestion.mordor_loader import MordorLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestMordorLoader:
    """Test suite for Mordor NDJSON evidence ingestion."""

    def test_load_powershell_httplistener(self):
        mordor_file = PROJECT_ROOT / "data" / "raw" / "mordor" / "psh_powershell_httplistener_2020-11-0204130683.json"
        assert mordor_file.exists(), "Mordor powershell file must exist"

        loader = MordorLoader(max_records=20)
        records = loader.load_file(
            mordor_file,
            source_dataset="mordor",
            attck_technique="T1059.001",
            tactic="Execution",
        )

        assert len(records) == 20
        rec = records[0]
        assert rec.modality == "json"
        assert rec.source_dataset == "mordor"
        assert rec.attck_technique == "T1059.001"
        assert rec.metadata["tactic"] == "Execution"
        assert "Event ID:" in rec.content
        assert rec.raw_reference == str(mordor_file)

    def test_load_psexec_lsa_secrets_dump(self):
        mordor_file = PROJECT_ROOT / "data" / "raw" / "mordor" / "cmd_psexec_lsa_secrets_dump_2020-10-1903305471.json"
        assert mordor_file.exists()

        loader = MordorLoader(max_records=10)
        records = loader.load_file(
            mordor_file,
            attck_technique="T1003",
            tactic="Credential Access",
        )

        assert len(records) == 10
        assert all(r.attck_technique == "T1003" for r in records)
        assert all(r.evidence_id.startswith("MORDOR-") for r in records)

    def test_nonexistent_file_raises_error(self):
        loader = MordorLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("nonexistent.json")
