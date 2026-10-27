"""EVTX parser for ingesting Windows Event Logs into CanonicalEvidenceRecords.

Uses python-evtx to parse binary .evtx files, extracts System and EventData
attributes, and produces canonical, human-readable representations suitable
for embedding, vector search, and LLM investigation.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from src.ingestion.evidence_schema import CanonicalEvidenceRecord, ModalityType

logger = logging.getLogger(__name__)


class EVTXParser:
    """Parses binary Windows Event Log (.evtx) files into CanonicalEvidenceRecords."""

    def __init__(self, max_records: int | None = None) -> None:
        """Initialize the EVTX parser.

        Args:
            max_records: Maximum number of records to parse per file (None for unlimited).
        """
        self.max_records = max_records

    def parse_file(
        self,
        file_path: str | Path,
        source_dataset: str = "atomic_evtx",
        attck_technique: str | None = None,
    ) -> list[CanonicalEvidenceRecord]:
        """Parse an EVTX file into a list of CanonicalEvidenceRecord instances.

        Args:
            file_path: Path to the .evtx file.
            source_dataset: Name of the originating dataset.
            attck_technique: Optional ATT&CK technique mapping if known.

        Returns:
            List of parsed CanonicalEvidenceRecord objects.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"EVTX file not found: {file_path}")

        try:
            import Evtx.Evtx as evtx
        except ImportError:
            raise ImportError(
                "python-evtx is required. Install with: pip install python-evtx"
            )

        records: list[CanonicalEvidenceRecord] = []
        parsed_count = 0

        logger.info("Parsing EVTX file: %s", file_path)

        with evtx.Evtx(str(file_path)) as log:
            for record_idx, record in enumerate(log.records()):
                if self.max_records and parsed_count >= self.max_records:
                    break

                try:
                    xml_content = record.xml()
                    parsed_event = self._parse_event_xml(xml_content)

                    # Build canonical evidence record
                    event_id = parsed_event.get("event_id", "Unknown")
                    provider = parsed_event.get("provider", "Unknown")
                    timestamp = parsed_event.get("time_created")

                    evidence_id = f"EVTX-{file_path.stem}-{record_idx + 1:04d}"
                    content = self._format_canonical_content(parsed_event, file_path.name)

                    rec = CanonicalEvidenceRecord(
                        evidence_id=evidence_id,
                        source_file=file_path.name,
                        source_dataset=source_dataset,
                        modality=ModalityType.EVTX.value,
                        content=content,
                        raw_reference=str(file_path),
                        metadata=parsed_event,
                        timestamp=timestamp,
                        event_type=f"EventID_{event_id}",
                        chunk_id=record_idx,
                        attck_technique=attck_technique,
                    )
                    records.append(rec)
                    parsed_count += 1
                except Exception as e:
                    logger.debug("Failed to parse record %d in %s: %s", record_idx, file_path.name, e)

        logger.info("Successfully parsed %d EVTX records from %s", len(records), file_path.name)
        return records

    def _parse_event_xml(self, xml_str: str) -> dict[str, Any]:
        """Parse XML string from an EVTX record into a structured dictionary."""
        # Strip XML namespaces for simplified element access
        clean_xml = re.sub(r'\sxmlns="[^"]+"', "", xml_str, count=1)
        root = ET.fromstring(clean_xml)

        data: dict[str, Any] = {}

        # Parse <System> elements
        system = root.find("System")
        if system is not None:
            provider_elem = system.find("Provider")
            if provider_elem is not None:
                data["provider"] = provider_elem.get("Name", "")
                data["provider_guid"] = provider_elem.get("Guid", "")

            event_id_elem = system.find("EventID")
            if event_id_elem is not None:
                data["event_id"] = event_id_elem.text or ""

            channel_elem = system.find("Channel")
            if channel_elem is not None:
                data["channel"] = channel_elem.text or ""

            computer_elem = system.find("Computer")
            if computer_elem is not None:
                data["computer"] = computer_elem.text or ""

            time_elem = system.find("TimeCreated")
            if time_elem is not None:
                data["time_created"] = time_elem.get("SystemTime", "")

            record_id_elem = system.find("EventRecordID")
            if record_id_elem is not None:
                data["record_id"] = record_id_elem.text or ""

            execution_elem = system.find("Execution")
            if execution_elem is not None:
                data["process_id"] = execution_elem.get("ProcessID", "")
                data["thread_id"] = execution_elem.get("ThreadID", "")

        # Parse <EventData> elements (Name-Value pairs)
        event_data = root.find("EventData")
        if event_data is not None:
            for data_elem in event_data.findall("Data"):
                name = data_elem.get("Name")
                text = data_elem.text or ""
                if name:
                    data[name] = text
                elif text:
                    data.setdefault("RawData", []).append(text)

        # Parse <UserData> if present
        user_data = root.find("UserData")
        if user_data is not None:
            for child in user_data:
                for subchild in child:
                    data[subchild.tag] = subchild.text or ""

        return data

    def _format_canonical_content(self, event: dict[str, Any], filename: str) -> str:
        """Format an event dictionary into canonical text for embedding and retrieval."""
        lines = [
            f"[WINDOWS EVENT LOG] Source: {filename}",
            f"Event ID: {event.get('event_id', 'Unknown')} | Provider: {event.get('provider', 'Unknown')}",
            f"Timestamp: {event.get('time_created', 'Unknown')} | Computer: {event.get('computer', 'Unknown')}",
            f"Channel: {event.get('channel', 'Unknown')}",
        ]

        # Highlight primary forensic fields if present
        forensic_fields = [
            ("Image", "Executable Image"),
            ("CommandLine", "Command Line"),
            ("ParentImage", "Parent Process"),
            ("ParentCommandLine", "Parent Command Line"),
            ("User", "Security User"),
            ("TargetFilename", "Target File"),
            ("SourceIp", "Source IP"),
            ("SourcePort", "Source Port"),
            ("DestinationIp", "Destination IP"),
            ("DestinationPort", "Destination Port"),
            ("DestinationHostname", "Destination Hostname"),
            ("Hashes", "Process Hashes"),
            ("TargetObject", "Registry Target"),
            ("Details", "Event Details"),
            ("RuleName", "Detection Rule"),
        ]

        lines.append("Forensic Details:")
        found_details = False
        for key, label in forensic_fields:
            if key in event and event[key]:
                lines.append(f"  - {label}: {event[key]}")
                found_details = True

        # Append remaining attributes
        excluded = {"event_id", "provider", "provider_guid", "channel", "computer",
                    "time_created", "record_id", "process_id", "thread_id"}
        excluded.update(k for k, _ in forensic_fields)

        remaining = {k: v for k, v in event.items() if k not in excluded and v}
        if remaining:
            lines.append("Additional Attributes:")
            for k, v in list(remaining.items())[:10]:
                lines.append(f"  - {k}: {v}")

        return "\n".join(lines)
