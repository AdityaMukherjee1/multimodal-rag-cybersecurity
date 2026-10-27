"""NVD / CVE vulnerability knowledge indexer.

Parses NIST NVD CVE JSON 2.0 vulnerability feeds, extracts CVSS v3.1 metrics,
CWE classifications, affected software configurations (CPEs), and technical descriptions,
and builds an extensible lookup and retrieval index.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.ingestion.evidence_schema import CanonicalEvidenceRecord, ModalityType

logger = logging.getLogger(__name__)


class CVEKnowledgeBase:
    """Provides indexed lookups and canonical evidence records for CVE vulnerabilities."""

    def __init__(self, index_path: str | Path | None = None) -> None:
        """Initialize the CVE knowledge base.

        Args:
            index_path: Optional path to a pre-built JSON index file.
        """
        self.cves: dict[str, dict[str, Any]] = {}
        self.index_loaded: bool = False

        if index_path and Path(index_path).exists():
            self.load_index(index_path)

    def build_from_nvd(
        self,
        nvd_file_path: str | Path,
        output_index_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Parse NVD CVE JSON 2.0 feed and build a structured vulnerability index.

        Args:
            nvd_file_path: Path to NVD JSON file.
            output_index_path: Optional destination to persist compiled index.

        Returns:
            Dictionary containing compiled CVE records.
        """
        nvd_path = Path(nvd_file_path)
        if not nvd_path.exists():
            raise FileNotFoundError(f"NVD JSON file not found: {nvd_path}")

        logger.info("Parsing NVD CVE records from %s ...", nvd_path)
        with open(nvd_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vulnerabilities = data.get("vulnerabilities", [])
        logger.info("Found %d vulnerability items in feed.", len(vulnerabilities))

        self.cves = {}

        for vuln in vulnerabilities:
            cve_item = vuln.get("cve", {})
            cve_id = cve_item.get("id", "").strip().upper()
            if not cve_id:
                continue

            # 1. English description
            descriptions = cve_item.get("descriptions", [])
            desc_text = ""
            for desc in descriptions:
                if desc.get("lang") == "en":
                    desc_text = desc.get("value", "")
                    break
            if not desc_text and descriptions:
                desc_text = descriptions[0].get("value", "")

            # 2. CVSS Metrics (Prioritize v3.1, fallback to v3.0, v2.0)
            metrics = cve_item.get("metrics", {})
            cvss_data: dict[str, Any] = {}

            if "cvssMetricV31" in metrics and metrics["cvssMetricV31"]:
                primary_v31 = metrics["cvssMetricV31"][0].get("cvssData", {})
                cvss_data = {
                    "version": "3.1",
                    "base_score": primary_v31.get("baseScore"),
                    "severity": primary_v31.get("baseSeverity", "UNKNOWN"),
                    "vector_string": primary_v31.get("vectorString", ""),
                    "attack_vector": primary_v31.get("attackVector"),
                    "attack_complexity": primary_v31.get("attackComplexity"),
                    "privileges_required": primary_v31.get("privilegesRequired"),
                    "user_interaction": primary_v31.get("userInteraction"),
                }
            elif "cvssMetricV30" in metrics and metrics["cvssMetricV30"]:
                primary_v30 = metrics["cvssMetricV30"][0].get("cvssData", {})
                cvss_data = {
                    "version": "3.0",
                    "base_score": primary_v30.get("baseScore"),
                    "severity": primary_v30.get("baseSeverity", "UNKNOWN"),
                    "vector_string": primary_v30.get("vectorString", ""),
                }

            # 3. CWE Weaknesses
            cwes: list[str] = []
            for weakness in cve_item.get("weaknesses", []):
                for desc in weakness.get("description", []):
                    cwe_val = desc.get("value")
                    if cwe_val and cwe_val not in ("NVD-CWE-noinfo", "NVD-CWE-Other"):
                        cwes.append(cwe_val)

            # 4. References
            references = [
                ref.get("url") for ref in cve_item.get("references", []) if ref.get("url")
            ][:5]

            # 5. Affected CPEs (Configurations)
            cpes: list[str] = []
            for config in cve_item.get("configurations", []):
                for node in config.get("nodes", []):
                    for cpe_match in node.get("cpeMatch", []):
                        criteria = cpe_match.get("criteria")
                        if criteria:
                            cpes.append(criteria)

            self.cves[cve_id] = {
                "cve_id": cve_id,
                "description": desc_text,
                "cvss": cvss_data,
                "cwes": cwes,
                "cpes": cpes[:10],
                "references": references,
                "published": cve_item.get("published"),
                "last_modified": cve_item.get("lastModified"),
            }

        self.index_loaded = True
        compiled_data = {
            "version": "NVD CVE 2.0 Compiled Index",
            "cve_count": len(self.cves),
            "cves": self.cves,
        }

        if output_index_path:
            out_p = Path(output_index_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(compiled_data, f, indent=2)
            logger.info("Compiled CVE index saved to %s (%d CVEs)", out_p, len(self.cves))

        return compiled_data

    def load_index(self, index_path: str | Path) -> None:
        """Load a pre-compiled CVE index.

        Args:
            index_path: Path to the JSON index file.
        """
        p = Path(index_path)
        if not p.exists():
            raise FileNotFoundError(f"CVE index not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.cves = data.get("cves", {})
        self.index_loaded = True
        logger.info("Loaded CVE index with %d CVEs from %s", len(self.cves), p.name)

    def get_cve(self, cve_id: str) -> dict[str, Any] | None:
        """Retrieve vulnerability details by CVE ID (e.g. 'CVE-2021-44228').

        Args:
            cve_id: Common Vulnerabilities and Exposures ID.

        Returns:
            CVE data dictionary or None if not indexed.
        """
        return self.cves.get(cve_id.strip().upper())

    def search_cves(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search CVE records by keyword in description or ID.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of matching CVE records.
        """
        q = query.lower().strip()
        tokens = [t for t in re.findall(r"[\w-]+", q) if len(t) > 1]
        results: list[dict[str, Any]] = []

        for cve in self.cves.values():
            cid = cve.get("cve_id", "").lower()
            desc = cve.get("description", "").lower()
            combined = f"{cid} {desc}"

            if q in combined or (tokens and all(t in combined for t in tokens)):
                results.append(cve)
                if len(results) >= limit:
                    break

        return results

    def to_canonical_records(
        self,
        source_reference: str = "data/raw/nvd/nvd_cve_sample.json",
    ) -> list[CanonicalEvidenceRecord]:
        """Convert all indexed CVEs into CanonicalEvidenceRecords for vector search.

        Args:
            source_reference: Original NVD file path reference.

        Returns:
            List of CanonicalEvidenceRecord objects.
        """
        records: list[CanonicalEvidenceRecord] = []

        for idx, (cid, cve) in enumerate(self.cves.items()):
            cvss = cve.get("cvss", {})
            score = cvss.get("base_score", "N/A")
            severity = cvss.get("severity", "N/A")
            vector = cvss.get("vector_string", "N/A")
            cwes_str = ", ".join(cve.get("cwes", [])) or "None specified"

            content = (
                f"[VULNERABILITY KNOWLEDGE: NVD / CVE] {cid}\n"
                f"CVSS v3.1 Severity: {severity} (Base Score: {score})\n"
                f"CVSS Vector String: {vector}\n"
                f"Weakness Classifications (CWE): {cwes_str}\n"
                f"Vulnerability Description:\n{cve.get('description', '')}\n"
                f"Published: {cve.get('published', 'Unknown')} | Last Modified: {cve.get('last_modified', 'Unknown')}"
            )

            rec = CanonicalEvidenceRecord(
                evidence_id=f"CVE-{cid.replace('-', '_')}",
                source_file="nvd_cve_sample.json",
                source_dataset="nvd",
                modality=ModalityType.NVD_CVE.value,
                content=content,
                raw_reference=source_reference,
                metadata={
                    "cve_id": cid,
                    "cvss_score": score,
                    "cvss_severity": severity,
                    "cwes": cve.get("cwes", []),
                    "published": cve.get("published"),
                },
                cve_id=cid,
                event_type="vulnerability_record",
                chunk_id=idx,
            )
            records.append(rec)

        logger.info("Generated %d canonical CVE evidence records.", len(records))
        return records
