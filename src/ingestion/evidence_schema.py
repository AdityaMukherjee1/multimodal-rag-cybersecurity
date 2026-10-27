"""Canonical Evidence Schema for Cybersecurity Incident Investigation.

This module defines the central data contract for all evidence ingested
into the CyberAI-Investigator research system. Every piece of raw evidence—
whether a Sysmon EVTX event, network flow CSV row, PDF advisory paragraph,
DOCX table, image screenshot, or MITRE/NVD entry—is mapped into this canonical
schema to guarantee strict provenance, traceability, and reproducible research.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class ModalityType(str, Enum):
    """Supported evidence modality types."""

    TEXT = "text"
    LOG = "log"
    JSON = "json"
    CSV = "csv"
    EVTX = "evtx"
    NETWORK_FLOW = "network_flow"
    PDF = "pdf"
    DOCX = "docx"
    IMAGE_SCREENSHOT = "image_screenshot"
    IMAGE_DIAGRAM = "image_diagram"
    IMAGE_TABLE = "image_table"
    MITRE_ATTACK = "mitre_attack"
    NVD_CVE = "nvd_cve"


@dataclass
class CanonicalEvidenceRecord:
    """Canonical representation of a single unit of cybersecurity evidence.

    Attributes:
        evidence_id: Unique identifier for this evidence item (e.g. 'EVT-2024-0001').
        source_file: Original filename from which this evidence was extracted.
        source_dataset: Originating dataset (e.g. 'atomic_evtx', 'mordor', 'cic_ids2017').
        modality: Modality category of the evidence.
        content: Canonical textual representation used for embedding and LLM reasoning.
        raw_reference: File path or URI pointing to the unmodified original evidence file.
        metadata: Dictionary containing domain-specific attributes (e.g. ProcessId, IPs).
        timestamp: ISO 8601 timestamp associated with the event (if known).
        incident_id: Correlation identifier linking multiple evidence records.
        event_type: Classification of the event (e.g. 'ProcessCreate', 'DDoS_Flow').
        page_number: Page number if extracted from a paginated document (PDF).
        chunk_id: Chunk sequence number if fragmented from a larger source.
        attck_technique: MITRE ATT&CK technique ID (e.g. 'T1059.001') if mapped.
        cve_id: Common Vulnerabilities and Exposures ID (e.g. 'CVE-2021-44228') if applicable.
        confidence_score: Assessment confidence or retrieval score (0.0 to 1.0).
        created_at: ISO timestamp when this canonical record was generated.
    """

    evidence_id: str
    source_file: str
    source_dataset: str
    modality: str
    content: str
    raw_reference: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    incident_id: str | None = None
    event_type: str | None = None
    page_number: int | None = None
    chunk_id: int | None = None
    attck_technique: str | None = None
    cve_id: str | None = None
    confidence_score: float | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CanonicalEvidenceRecord:
        """Instantiate a CanonicalEvidenceRecord from a dictionary."""
        return cls(**data)

    def to_json(self, indent: int | None = None) -> str:
        """Serialize the record to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> CanonicalEvidenceRecord:
        """Deserialize a CanonicalEvidenceRecord from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_document_dict(self) -> dict[str, Any]:
        """Convert to a dictionary compatible with the Phase 1 VectorStore.

        Maintains backward compatibility with Phase 1 vector storage.
        """
        meta = {
            **self.metadata,
            "evidence_id": self.evidence_id,
            "source_file": self.source_file,
            "source_dataset": self.source_dataset,
            "modality": self.modality,
            "raw_reference": self.raw_reference,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "attck_technique": self.attck_technique,
            "cve_id": self.cve_id,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "filename": Path(self.source_file).name,
        }
        return {
            "content": self.content,
            "metadata": meta,
        }
