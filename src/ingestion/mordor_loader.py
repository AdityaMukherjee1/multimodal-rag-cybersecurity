"""Mordor structured NDJSON evidence loader.

Parses simulated adversary security event logs from the OTRF Mordor project,
preserving event fields, timestamps, hostnames, and explicit dataset-level
MITRE ATT&CK technique attributions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.ingestion.evidence_schema import CanonicalEvidenceRecord, ModalityType

logger = logging.getLogger(__name__)


class MordorLoader:
    """Loads and converts Mordor NDJSON event datasets into CanonicalEvidenceRecords."""

    def __init__(self, max_records: int | None = None) -> None:
        """Initialize the Mordor loader.

        Args:
            max_records: Maximum number of events to load per file (None for unlimited).
        """
        self.max_records = max_records

    def load_file(
        self,
        file_path: str | Path,
        source_dataset: str = "mordor",
        attck_technique: str | None = None,
        tactic: str | None = None,
    ) -> list[CanonicalEvidenceRecord]:
        """Load an NDJSON file and return CanonicalEvidenceRecord objects.

        Args:
            file_path: Path to the .json (NDJSON) file.
            source_dataset: Dataset category name.
            attck_technique: Dataset-level ATT&CK technique mapping (e.g. 'T1059.001').
            tactic: Dataset-level ATT&CK tactic (e.g. 'Execution', 'Credential Access').

        Returns:
            List of CanonicalEvidenceRecord objects.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Mordor file not found: {file_path}")

        records: list[CanonicalEvidenceRecord] = []
        logger.info("Loading Mordor NDJSON data: %s", file_path.name)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                if self.max_records is not None and len(records) >= self.max_records:
                    break

                try:
                    event = json.loads(line)
                except Exception as e:
                    logger.debug("Skipping unparseable JSON on line %d: %s", line_idx + 1, e)
                    continue

                event_id = event.get("EventID", event.get("event_id", "Unknown"))
                source_name = event.get("SourceName", event.get("source_name", "Unknown"))
                hostname = event.get("Hostname", event.get("hostname", "Unknown"))
                channel = event.get("Channel", event.get("channel", "Security"))
                timestamp = event.get("TimeCreated", event.get("@timestamp", event.get("time_created")))
                message = event.get("Message", "").strip()

                evidence_id = f"MORDOR-{file_path.stem}-{len(records) + 1:05d}"

                content = self._format_canonical_content(
                    event=event,
                    filename=file_path.name,
                    record_idx=len(records) + 1,
                    event_id=event_id,
                    source_name=source_name,
                    hostname=hostname,
                    channel=channel,
                    timestamp=timestamp,
                    message=message,
                    attck_technique=attck_technique,
                    tactic=tactic,
                )

                rec = CanonicalEvidenceRecord(
                    evidence_id=evidence_id,
                    source_file=file_path.name,
                    source_dataset=source_dataset,
                    modality=ModalityType.JSON.value,
                    content=content,
                    raw_reference=str(file_path),
                    metadata={
                        "event_id": str(event_id),
                        "source_name": source_name,
                        "hostname": hostname,
                        "channel": channel,
                        "task": event.get("Task"),
                        "level": event.get("Level"),
                        "tactic": tactic,
                        "dataset_technique": attck_technique,
                    },
                    timestamp=str(timestamp) if timestamp else None,
                    event_type=f"EventID_{event_id}",
                    chunk_id=len(records),
                    attck_technique=attck_technique,
                )
                records.append(rec)

        logger.info(
            "Extracted %d Mordor evidence records from %s",
            len(records), file_path.name,
        )
        return records

    def _format_canonical_content(
        self,
        event: dict[str, Any],
        filename: str,
        record_idx: int,
        event_id: Any,
        source_name: str,
        hostname: str,
        channel: str,
        timestamp: Any,
        message: str,
        attck_technique: str | None,
        tactic: str | None,
    ) -> str:
        """Format event dictionary into canonical text representation."""
        lines = [
            f"[MORDOR SIMULATED ADVERSARY EVENT] Source: {filename} | Event #{record_idx}",
            f"Event ID: {event_id} | Provider: {source_name} | Channel: {channel}",
            f"Host: {hostname} | Timestamp: {timestamp or 'Unknown'}",
        ]

        if attck_technique or tactic:
            lines.append(
                f"Adversary Emulation Context: Tactic='{tactic or 'Unknown'}' | ATT&CK='{attck_technique or 'Unknown'}'"
            )

        if message:
            # Clean excessive whitespace inside message
            clean_msg = "\n".join(
                line.strip() for line in message.splitlines() if line.strip()
            )
            lines.append(f"Event Log Message:\n{clean_msg}")
        else:
            # If no formatted Message, output top key attributes
            lines.append("Event Data Attributes:")
            for k, v in list(event.items())[:8]:
                if k not in ("EventID", "SourceName", "Hostname", "Channel", "TimeCreated", "@timestamp"):
                    lines.append(f"  - {k}: {v}")

        return "\n".join(lines)
