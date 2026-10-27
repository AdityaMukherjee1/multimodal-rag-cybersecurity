# CyberAI-Investigator Research Data Layer

This directory houses the multimodal research data foundation for the undergraduate research project:
**“Evidence-Grounded Multimodal RAG for Cybersecurity Incident Investigation”**

---

## 1. Architectural Philosophy

Our design enforces a strict separation of concerns across data tiers:

```text
AUTHENTIC PUBLIC EVIDENCE (data/raw/)
        │
        ▼ (Modality-Specific Parsers: EVTX, Network CSV, PDF, DOCX, Images)
NORMALIZED MACHINE-READABLE RECORDS (data/processed/ & CanonicalEvidenceRecord)
        │
        ▼ (Embedding Models: BGE-small / Multimodal Vision Embeddings)
VECTOR STORE & EVIDENCE PROVENANCE (FAISS + Synchronized Metadata)
        │
        ▼
EVIDENCE-GROUNDED RETRIEVAL & VERIFICATION
```

### Core Principles

1. **Raw Evidence Preservation**: Files stored in `data/raw/` are untouched, authentic original artifacts. They are never modified in-place or deleted during processing.
2. **Canonical Evidence Schema (`CanonicalEvidenceRecord`)**: Every evidence item, whether an EVTX event, a network flow row, or a DOCX table, is normalized into a standard dataclass with mandatory provenance:
   - `evidence_id`: Unique persistent identifier.
   - `raw_reference`: Path to the source file.
   - `modality`: Modality category (`evtx`, `network_flow`, `pdf`, `docx`, `image_screenshot`, etc.).
   - `content`: Canonical textual summary suitable for semantic embedding and LLM reasoning.
   - `metadata`: Forensics details (Process ID, IPs, Ports, hashes, CVSS scores).
   - `attck_technique`: MITRE ATT&CK technique mapping (e.g. `T1059.001`, `T1003.001`).
3. **No Database Monoliths**: The vector database does **not** store the sole copy of evidence. It stores vectors, chunk IDs, and metadata that link back to the canonical processed records and raw artifacts.

---

## 2. Directory Layout

```text
data/
├── raw/                               # Unmodified original evidence artifacts
│   ├── atomic_evtx/                   # Windows Event Logs (.evtx) from Atomic Red Team & DFIR
│   ├── mordor/                        # OTRF Mordor simulated adversary telemetry (NDJSON)
│   ├── cic_ids2017/                   # Network intrusion flow telemetry (CSV)
│   ├── mitre_attack/                  # Official MITRE Enterprise ATT&CK (STIX 2.1 JSON)
│   ├── nvd/                           # Official NIST NVD CVE vulnerability records (JSON)
│   ├── documents/                     # Textual incident reports & advisories
│   │   ├── pdf/                       # NIST SP 800-61r2 & CISA joint advisories
│   │   └── docx/                      # SOC Incident Investigation Briefing with IOC tables
│   └── images/                        # Visual evidence
│       ├── screenshots/               # SIEM alerts & terminal forensic captures
│       ├── diagrams/                  # Attack flow & network topology diagrams
│       └── tables/                    # Visual threat intelligence IOC matrices
│
├── processed/                         # Normalized machine-readable representations
│   ├── text/
│   ├── logs/
│   ├── evtx/
│   ├── network/
│   ├── documents/
│   ├── images/
│   ├── mitre_attack/
│   └── nvd/
│
├── sample/                            # Lightweight synthetic samples used for unit tests
│   ├── sample_sysmon_logs.json
│   ├── sample_incident_report.txt
│   └── sample_firewall_logs.csv
│
├── dataset_manifest.yaml              # Verified dataset manifest and provenance records
└── README.md                          # This documentation
```

---

## 3. Dataset Summary

| Dataset / Collection | Primary Modality | Format | Authoritative Source | Primary Forensic Role |
|---|---|---|---|---|
| **Atomic-EVTX** | Host Event Log | `.evtx`, `.csv` | Samir Bousseaden & Arda Büyükkaya | Windows process creation, DCSync, credential dumping |
| **Mordor** | Host Event Log | `.json` (NDJSON) | OTRF (Roberto & Jose Luis Rodriguez) | ATT&CK-aligned adversary emulation (PowerShell, PsExec, Seatbelt) |
| **CIC-IDS2017** | Network Telemetry | `.csv` (78 features) | Canadian Institute for Cybersecurity (UNB) | Network flow analysis, DoS/DDoS detection, port scanning |
| **MITRE ATT&CK** | Cybersecurity Knowledge | `.json` (STIX 2.1) | MITRE Corporation | Tactics, Techniques, Mitigations, and Threat Groups |
| **NVD CVE** | Vulnerability Knowledge | `.json` (NVD 2.0) | NIST NVD API 2.0 | CVSS scores, affected CPEs, exploit vulnerability context |
| **Incident Documents** | Document Evidence | `.pdf`, `.docx` | NIST / CISA / SOC Incident Briefing | High-level investigation guidelines, timelines, and IOC tables |
| **Visual Evidence** | Multimodal Images | `.png` | SIEM, Terminal, Topology Diagrams | Cross-modal reasoning, visual alert verification |

---

## 4. Inventorying Datasets

Run the automated data inventory script at any time to inspect dataset statistics:

```bash
python scripts/inventory_datasets.py
```
