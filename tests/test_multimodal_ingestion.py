"""Unit tests for multimodal cybersecurity evidence parsers."""

from pathlib import Path
import pytest

from src.ingestion.evtx_parser import EVTXParser
from src.ingestion.docx_loader import DOCXLoader
from src.ingestion.network_loader import NetworkFlowLoader
from src.ingestion.image_metadata_loader import ImageEvidenceLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestEVTXParser:
    """Tests for the EVTX log parser."""

    def test_parse_real_evtx_file(self):
        evtx_path = PROJECT_ROOT / "data" / "raw" / "atomic_evtx" / "Exec_sysmon_meterpreter_reversetcp_msipackage.evtx"
        assert evtx_path.exists(), "EVTX file must exist in data/raw/atomic_evtx"

        parser = EVTXParser(max_records=10)
        records = parser.parse_file(evtx_path, source_dataset="atomic_evtx", attck_technique="T1059")

        assert len(records) > 0
        rec = records[0]
        assert rec.modality == "evtx"
        assert rec.source_file == evtx_path.name
        assert rec.attck_technique == "T1059"
        assert "Event ID:" in rec.content
        assert "provider" in rec.metadata

    def test_evtx_nonexistent_file(self):
        parser = EVTXParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file("nonexistent.evtx")


class TestDOCXLoader:
    """Tests for the DOCX document loader."""

    def test_load_real_docx_file(self):
        docx_path = PROJECT_ROOT / "data" / "raw" / "documents" / "docx" / "soc_incident_investigation_briefing.docx"
        assert docx_path.exists(), "DOCX file must exist in data/raw/documents/docx"

        loader = DOCXLoader()
        records = loader.load_file(docx_path)

        assert len(records) >= 2
        modalities = [r.modality for r in records]
        assert all(m == "docx" for m in modalities)

        # Check that table was parsed as markdown
        table_records = [r for r in records if r.event_type == "document_table"]
        assert len(table_records) >= 1
        assert "| Artifact Type |" in table_records[0].content


class TestNetworkFlowLoader:
    """Tests for the network flow loader."""

    def test_load_real_cicids_sample(self):
        csv_path = PROJECT_ROOT / "data" / "raw" / "cic_ids2017" / "cicids2017_representative_sample.csv"
        assert csv_path.exists(), "CIC-IDS2017 CSV must exist in data/raw/cic_ids2017"

        loader = NetworkFlowLoader(max_records=20)
        records = loader.load_file(csv_path)

        assert len(records) == 20
        rec = records[0]
        assert rec.modality == "network_flow"
        assert "Traffic Classification:" in rec.content
        assert "Destination Port:" in rec.content
        assert "label" in rec.metadata


class TestImageEvidenceLoader:
    """Tests for the image evidence loader."""

    def test_load_real_image(self):
        img_path = PROJECT_ROOT / "data" / "raw" / "images" / "screenshots" / "siem_alert_powershell_execution.png"
        assert img_path.exists(), "Image must exist in data/raw/images/screenshots"

        loader = ImageEvidenceLoader()
        record = loader.load_file(
            img_path,
            modality="image_screenshot",
            source_dataset="siem_screenshots",
            description="SIEM alert capture",
            attck_technique="T1059.001",
        )

        assert record.modality == "image_screenshot"
        assert record.metadata["width"] == 900
        assert record.metadata["height"] == 480
        assert record.metadata["format"] == "PNG"
        assert record.attck_technique == "T1059.001"
        assert "Visual Content Description:" in record.content
