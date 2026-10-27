"""MITRE ATT&CK STIX knowledge loader and fast lookup index.

Parses Enterprise ATT&CK STIX 2.1 JSON, extracts techniques, tactics,
mitigations, software, and relationships, and compiles a lightweight
lookup index for low-latency retrieval during incident investigation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.ingestion.evidence_schema import CanonicalEvidenceRecord, ModalityType

logger = logging.getLogger(__name__)


class AttackKnowledgeBase:
    """Provides indexed lookups and canonical evidence records for MITRE ATT&CK."""

    def __init__(self, index_path: str | Path | None = None) -> None:
        """Initialize the ATT&CK knowledge base.

        Args:
            index_path: Optional path to a pre-built JSON index file.
        """
        self.techniques: dict[str, dict[str, Any]] = {}
        self.tactics: dict[str, dict[str, Any]] = {}
        self.mitigations: dict[str, dict[str, Any]] = {}
        self.software: dict[str, dict[str, Any]] = {}
        self.index_loaded: bool = False

        if index_path and Path(index_path).exists():
            self.load_index(index_path)

    def build_from_stix(
        self,
        stix_file_path: str | Path,
        output_index_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Parse raw STIX 2.1 JSON and construct a lightweight lookup index.

        Args:
            stix_file_path: Path to enterprise-attack.json.
            output_index_path: Optional path to persist the compiled index.

        Returns:
            Dictionary containing compiled techniques, tactics, mitigations, and software.
        """
        stix_path = Path(stix_file_path)
        if not stix_path.exists():
            raise FileNotFoundError(f"MITRE ATT&CK STIX file not found: {stix_path}")

        logger.info("Parsing MITRE ATT&CK STIX bundle from %s ...", stix_path)
        with open(stix_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        objects = bundle.get("objects", [])
        logger.info("Processing %d STIX objects...", len(objects))

        # Reset in-memory stores
        self.techniques = {}
        self.tactics = {}
        self.mitigations = {}
        self.software = {}

        stix_id_to_tech_id: dict[str, str] = {}
        relationships: list[dict[str, Any]] = []

        # First pass: collect entities
        for obj in objects:
            obj_type = obj.get("type")

            # 1. Attack Patterns (Techniques & Sub-techniques)
            if obj_type == "attack-pattern":
                if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
                    continue

                external_id = None
                url = None
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        external_id = ref.get("external_id")
                        url = ref.get("url")
                        break

                if not external_id:
                    continue

                tactics = [
                    phase.get("phase_name", "")
                    for phase in obj.get("kill_chain_phases", [])
                    if phase.get("kill_chain_name") == "mitre-attack"
                ]

                tech_data: dict[str, Any] = {
                    "technique_id": external_id,
                    "stix_id": obj.get("id"),
                    "name": obj.get("name", ""),
                    "description": obj.get("description", ""),
                    "tactics": tactics,
                    "platforms": obj.get("x_mitre_platforms", []),
                    "data_sources": obj.get("x_mitre_data_sources", []),
                    "detection": obj.get("x_mitre_detection", ""),
                    "is_subtechnique": obj.get("x_mitre_is_subtechnique", False),
                    "url": url,
                    "mitigations": [],
                    "software": [],
                }

                self.techniques[external_id.upper()] = tech_data
                stix_id_to_tech_id[obj.get("id")] = external_id.upper()

            # 2. Tactics
            elif obj_type == "x-mitre-tactic":
                shortname = obj.get("x_mitre_shortname", "")
                ext_id = None
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        ext_id = ref.get("external_id")
                        break

                self.tactics[shortname.lower()] = {
                    "tactic_id": ext_id,
                    "shortname": shortname,
                    "name": obj.get("name", ""),
                    "description": obj.get("description", ""),
                }

            # 3. Mitigations (Course of Action)
            elif obj_type == "course-of-action":
                if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
                    continue
                ext_id = None
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        ext_id = ref.get("external_id")
                        break

                self.mitigations[obj.get("id")] = {
                    "mitigation_id": ext_id,
                    "name": obj.get("name", ""),
                    "description": obj.get("description", ""),
                }

            # 4. Software & Malware
            elif obj_type in ("malware", "tool"):
                if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
                    continue
                ext_id = None
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack":
                        ext_id = ref.get("external_id")
                        break

                self.software[obj.get("id")] = {
                    "software_id": ext_id,
                    "name": obj.get("name", ""),
                    "type": obj_type,
                    "description": obj.get("description", ""),
                }

            # 5. Relationships
            elif obj_type == "relationship":
                relationships.append(obj)

        # Second pass: map relationships (mitigations and software to techniques)
        for rel in relationships:
            rel_type = rel.get("relationship_type")
            src_id = rel.get("source_ref")
            target_id = rel.get("target_ref")

            if target_id in stix_id_to_tech_id:
                tech_id = stix_id_to_tech_id[target_id]
                tech_entry = self.techniques[tech_id]

                if rel_type == "mitigates" and src_id in self.mitigations:
                    mit = self.mitigations[src_id]
                    tech_entry["mitigations"].append({
                        "id": mit.get("mitigation_id"),
                        "name": mit.get("name"),
                        "description": rel.get("description", mit.get("description")),
                    })

                elif rel_type == "uses" and src_id in self.software:
                    sw = self.software[src_id]
                    tech_entry["software"].append({
                        "id": sw.get("software_id"),
                        "name": sw.get("name"),
                        "type": sw.get("type"),
                    })

        self.index_loaded = True
        compiled_data = {
            "version": "ATT&CK STIX 2.1 Compiled Index",
            "technique_count": len(self.techniques),
            "tactic_count": len(self.tactics),
            "mitigation_count": len(self.mitigations),
            "software_count": len(self.software),
            "techniques": self.techniques,
            "tactics": self.tactics,
        }

        if output_index_path:
            out_p = Path(output_index_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(compiled_data, f, indent=2)
            logger.info(
                "Compiled ATT&CK index saved to %s (%d techniques)",
                out_p, len(self.techniques),
            )

        return compiled_data

    def load_index(self, index_path: str | Path) -> None:
        """Load a pre-compiled ATT&CK index for instant lookup.

        Args:
            index_path: Path to the JSON index file.
        """
        p = Path(index_path)
        if not p.exists():
            raise FileNotFoundError(f"ATT&CK index not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.techniques = data.get("techniques", {})
        self.tactics = data.get("tactics", {})
        self.index_loaded = True
        logger.info(
            "Loaded ATT&CK index with %d techniques from %s",
            len(self.techniques), p.name,
        )

    def get_technique(self, technique_id: str) -> dict[str, Any] | None:
        """Look up a technique by ID (e.g. 'T1059.001' or 't1059').

        Args:
            technique_id: ATT&CK technique identifier.

        Returns:
            Technique dictionary or None if not found.
        """
        return self.techniques.get(technique_id.strip().upper())

    def search_techniques(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search techniques by keyword in name or description.

        Args:
            query: Keyword search string.
            limit: Maximum results to return.

        Returns:
            List of matching technique dictionaries.
        """
        q = query.lower()
        results: list[dict[str, Any]] = []

        for tech in self.techniques.values():
            name = tech.get("name", "").lower()
            desc = tech.get("description", "").lower()
            tid = tech.get("technique_id", "").lower()

            if q in tid or q in name or q in desc:
                results.append(tech)
                if len(results) >= limit:
                    break

        return results

    def get_tactics(self) -> dict[str, dict[str, Any]]:
        """Return all tactic definitions."""
        return self.tactics

    def to_canonical_records(
        self,
        source_reference: str = "data/raw/mitre_attack/enterprise-attack.json",
    ) -> list[CanonicalEvidenceRecord]:
        """Convert all indexed techniques into CanonicalEvidenceRecords for vector search.

        Args:
            source_reference: Original STIX path reference.

        Returns:
            List of CanonicalEvidenceRecord objects.
        """
        records: list[CanonicalEvidenceRecord] = []

        for idx, (tid, tech) in enumerate(self.techniques.items()):
            tactics_str = ", ".join(tech.get("tactics", [])) or "Enterprise"
            platforms_str = ", ".join(tech.get("platforms", []))
            data_sources_str = ", ".join(tech.get("data_sources", []))

            mitigations_list = tech.get("mitigations", [])
            mitigation_text = "\n".join(
                f"  - {m.get('name')}: {m.get('description', '')[:120]}..."
                for m in mitigations_list[:3]
            ) if mitigations_list else "  - See MITRE ATT&CK guidance."

            content = (
                f"[MITRE ATT&CK KNOWLEDGE] Technique {tid}: {tech.get('name')}\n"
                f"Tactics / Kill Chain: {tactics_str}\n"
                f"Platforms: {platforms_str} | Data Sources: {data_sources_str}\n"
                f"Technical Description:\n{tech.get('description', '')}\n"
                f"Detection Strategy:\n{tech.get('detection', 'Monitor relevant telemetry.')}\n"
                f"Recommended Mitigations:\n{mitigation_text}"
            )

            rec = CanonicalEvidenceRecord(
                evidence_id=f"ATTCK-{tid.replace('.', '_')}",
                source_file="enterprise-attack.json",
                source_dataset="mitre_attack",
                modality=ModalityType.MITRE_ATTACK.value,
                content=content,
                raw_reference=source_reference,
                metadata={
                    "technique_id": tid,
                    "technique_name": tech.get("name"),
                    "tactics": tech.get("tactics", []),
                    "platforms": tech.get("platforms", []),
                    "url": tech.get("url"),
                    "is_subtechnique": tech.get("is_subtechnique", False),
                },
                attck_technique=tid,
                event_type="knowledge_technique",
                chunk_id=idx,
            )
            records.append(rec)

        logger.info("Generated %d canonical ATT&CK evidence records.", len(records))
        return records
