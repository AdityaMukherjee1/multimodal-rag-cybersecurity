"""Network flow evidence loader for tabular cybersecurity traffic data.

Parses network flow CSV files (e.g. CIC-IDS2017), normalizes column headers,
and generates canonical textual representations summarizing flow statistics,
packet counts, protocol details, flag characteristics, and intrusion labels.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from src.ingestion.evidence_schema import CanonicalEvidenceRecord, ModalityType

logger = logging.getLogger(__name__)


class NetworkFlowLoader:
    """Loads and transforms network flow CSV logs into CanonicalEvidenceRecords."""

    def __init__(self, max_records: int = 500) -> None:
        """Initialize the network loader.

        Args:
            max_records: Maximum number of flow records to extract per file.
        """
        self.max_records = max_records

    def load_file(
        self,
        file_path: str | Path,
        source_dataset: str = "cic_ids2017",
    ) -> list[CanonicalEvidenceRecord]:
        """Load network flow CSV and convert to CanonicalEvidenceRecord objects.

        Args:
            file_path: Path to the network flow CSV file.
            source_dataset: Originating dataset name.

        Returns:
            List of CanonicalEvidenceRecord objects.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Network flow file not found: {file_path}")

        records: list[CanonicalEvidenceRecord] = []
        logger.info("Loading network flow data: %s", file_path.name)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)

            for row_idx, raw_row in enumerate(reader):
                if row_idx >= self.max_records:
                    break

                row = {k.strip(): v.strip() for k, v in raw_row.items() if k is not None}

                label = row.get("Label", row.get("label", "UNKNOWN"))
                duration = row.get("Flow Duration", row.get("flow_duration", "0"))
                fwd_packets = row.get("Total Fwd Packets", row.get("total_fwd_packets", "0"))
                bwd_packets = row.get("Total Backward Packets", row.get("total_bwd_packets", "0"))
                fwd_len = row.get("Total Length of Fwd Packets", "0")
                bwd_len = row.get("Total Length of Bwd Packets", "0")
                flow_bytes = row.get("Flow Bytes/s", row.get("flow_bytes_s", "0"))
                flow_packets = row.get("Flow Packets/s", row.get("flow_packets_s", "0"))
                pkt_mean = row.get("Packet Length Mean", row.get("Average Packet Size", "0"))
                pkt_max = row.get("Max Packet Length", "0")

                # Handle Destination Port if present in CSV, otherwise clearly note its absence
                dst_port = row.get("Destination Port", row.get("dst_port"))
                port_str = dst_port if dst_port is not None else "N/A (Statistical Feature Set)"

                syn = row.get("SYN Flag Count", "0")
                ack = row.get("ACK Flag Count", "0")
                fin = row.get("FIN Flag Count", "0")
                rst = row.get("RST Flag Count", "0")
                psh = row.get("PSH Flag Count", row.get("Fwd PSH Flags", "0"))
                urg = row.get("URG Flag Count", row.get("Fwd URG Flags", "0"))

                evidence_id = f"NET-{file_path.stem}-{row_idx + 1:05d}"

                content = (
                    f"[NETWORK FLOW EVIDENCE] Source: {file_path.name} | Flow #{row_idx + 1}\n"
                    f"Traffic Classification: {label}\n"
                    f"Destination Port: {port_str} | Flow Duration: {duration} microseconds\n"
                    f"Packet Statistics: Forward Packets={fwd_packets} ({fwd_len} bytes), Backward Packets={bwd_packets} ({bwd_len} bytes)\n"
                    f"Flow Rates: {flow_bytes} Bytes/s, {flow_packets} Packets/s | Avg Packet Size: {pkt_mean} (Max: {pkt_max})\n"
                    f"TCP Flags: SYN={syn}, ACK={ack}, FIN={fin}, RST={rst}, PSH={psh}, URG={urg}"
                )

                metadata: dict[str, Any] = {
                    "label": label,
                    "destination_port": port_str,
                    "flow_duration": duration,
                    "total_fwd_packets": fwd_packets,
                    "total_bwd_packets": bwd_packets,
                    "flow_bytes_per_sec": flow_bytes,
                    "flow_packets_per_sec": flow_packets,
                    "packet_length_mean": pkt_mean,
                    "row_index": row_idx,
                }

                rec = CanonicalEvidenceRecord(
                    evidence_id=evidence_id,
                    source_file=file_path.name,
                    source_dataset=source_dataset,
                    modality=ModalityType.NETWORK_FLOW.value,
                    content=content,
                    raw_reference=str(file_path),
                    metadata=metadata,
                    event_type=f"NetworkFlow_{label}",
                    chunk_id=row_idx,
                )
                records.append(rec)

        logger.info(
            "Extracted %d network flow evidence records from %s",
            len(records), file_path.name,
        )
        return records
