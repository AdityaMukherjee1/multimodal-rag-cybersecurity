"""Unit tests for the CanonicalEvidenceRecord schema."""

import json
import pytest
from src.ingestion.evidence_schema import CanonicalEvidenceRecord, ModalityType


def test_canonical_evidence_record_creation():
    """Verify CanonicalEvidenceRecord instantiation and defaults."""
    rec = CanonicalEvidenceRecord(
        evidence_id="EVT-TEST-001",
        source_file="test_sysmon.evtx",
        source_dataset="atomic_evtx",
        modality=ModalityType.EVTX.value,
        content="Process Create: powershell.exe -ExecutionPolicy Bypass",
        raw_reference="data/raw/atomic_evtx/test_sysmon.evtx",
        metadata={"process_id": "4528", "user": "DOMAIN\\analyst"},
        timestamp="2024-01-15T08:23:14Z",
        event_type="ProcessCreate",
        attck_technique="T1059.001",
    )

    assert rec.evidence_id == "EVT-TEST-001"
    assert rec.source_file == "test_sysmon.evtx"
    assert rec.source_dataset == "atomic_evtx"
    assert rec.modality == "evtx"
    assert rec.event_type == "ProcessCreate"
    assert rec.attck_technique == "T1059.001"
    assert rec.metadata["process_id"] == "4528"
    assert rec.created_at is not None


def test_canonical_evidence_record_serialization():
    """Verify serialization to/from dictionary and JSON."""
    rec = CanonicalEvidenceRecord(
        evidence_id="NET-TEST-002",
        source_file="cicids2017.csv",
        source_dataset="cic_ids2017",
        modality=ModalityType.NETWORK_FLOW.value,
        content="Flow Duration: 4us, Total Fwd Packets: 2",
        raw_reference="data/raw/cic_ids2017/cicids2017.csv",
        metadata={"label": "BENIGN", "dst_port": "80"},
    )

    d = rec.to_dict()
    assert isinstance(d, dict)
    assert d["evidence_id"] == "NET-TEST-002"

    json_str = rec.to_json()
    assert isinstance(json_str, str)
    assert "NET-TEST-002" in json_str

    rec_recovered = CanonicalEvidenceRecord.from_json(json_str)
    assert rec_recovered.evidence_id == rec.evidence_id
    assert rec_recovered.modality == rec.modality
    assert rec_recovered.content == rec.content


def test_canonical_evidence_record_vector_store_compatibility():
    """Verify conversion to document dict compatible with VectorStore."""
    rec = CanonicalEvidenceRecord(
        evidence_id="DOC-TEST-003",
        source_file="soc_report.docx",
        source_dataset="documents_docx",
        modality=ModalityType.DOCX.value,
        content="Executive summary text...",
        raw_reference="data/raw/documents/docx/soc_report.docx",
        metadata={"heading": "Overview"},
        chunk_id=1,
    )

    doc_dict = rec.to_document_dict()
    assert "content" in doc_dict
    assert "metadata" in doc_dict
    assert doc_dict["content"] == "Executive summary text..."
    assert doc_dict["metadata"]["evidence_id"] == "DOC-TEST-003"
    assert doc_dict["metadata"]["filename"] == "soc_report.docx"
    assert doc_dict["metadata"]["chunk_id"] == 1
