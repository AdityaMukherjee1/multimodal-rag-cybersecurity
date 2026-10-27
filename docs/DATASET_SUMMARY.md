# Dataset Summary
## Evidence-Grounded Multimodal RAG for Cybersecurity Incident Investigation

All values measured directly from the filesystem on **2026-09-07**. No values are estimated or extrapolated.

---

## Core Research Datasets

| Dataset | Evidence Type | Modality | Authoritative Source | License | Size | Records / Objects | Main Research Use |
|---|---|---|---|---|---|---|---|
| **Atomic-EVTX** | Host telemetry | `.evtx` binary | sbousseaden/EVTX-ATTACK-SAMPLES (MIT); arniki/atomic-evtx (Apache 2.0) | MIT / Apache 2.0 | ~418 KB | 52 events across 5 files | Process execution, DCSync, LSASS access, persistence detection |
| **Mordor** | Security events | `.json` (NDJSON) | OTRF/Security-Datasets (R. Rodriguez) | CC BY-SA 4.0 | ~1.45 MB | 847 events (3 datasets) | ATT&CK-aligned simulation: PowerShell exec, PsExec, Seatbelt discovery |
| **CIC-IDS2017** | Network flows | `.csv` | Canadian Institute for Cybersecurity (UNB) / Western OC2 Lab mirror | Open academic use | 1.12 MB | 3,000 flow rows, 78 columns | Network intrusion detection: DoS (89%), BENIGN (11%) |
| **MITRE ATT&CK** | Threat knowledge | `.json` (STIX 2.1) | MITRE Corporation / CTID | Apache 2.0 | 51.34 MB | 26,086 STIX objects (858 attack-patterns) | Technique lookup, tactic context, adversary group mapping |
| **NVD CVE** | Vulnerabilities | `.json` (NVD 2.0) | NIST NVD API 2.0 | U.S. Government Work (Public Domain) | 277 KB | 7 CVE records | CVSS severity, exploit context, vulnerability correlation |

---

## Multimodal Document Datasets

| Dataset | Evidence Type | Modality | Source | License | Size | Content |
|---|---|---|---|---|---|---|
| **NIST SP 800-61r2** | Incident handling guide | `.pdf` (80 pages) | NIST Special Publications | U.S. Government Work | 1.66 MB | Text + diagrams; authoritative IR methodology |
| **CISA Advisory AA23-347A** | Threat advisory | `.pdf` (1 page) | Generated from CISA advisory text | Research use | 1.9 KB | Text-only; SVR TeamCity exploitation context |
| **SOC Incident Briefing** | Incident report | `.docx` | Generated document | Research use | 36.8 KB | 8 paragraphs + 1 IOC table (6 rows × 4 cols) |

---

## Visual Evidence Datasets

| File | Type | Format | Dimensions | File Size | Semantic Content | VLM Ready |
|---|---|---|---|---|---|---|
| `siem_alert_powershell_execution.png` | Screenshot | PNG/RGB | 900×480 | 28.9 KB | SIEM alert: PowerShell bypass | ❌ Phase 3 |
| `terminal_credential_access_mimikatz.png` | Screenshot | PNG/RGB | 850×420 | 22.6 KB | Terminal: Mimikatz sekurlsa | ❌ Phase 3 |
| `incident_attack_flow_diagram.png` | Diagram | PNG/RGB | 960×540 | 26.1 KB | Attack flow: TA → Exec → Persist → C2 | ❌ Phase 3 |
| `visual_threat_intel_ioc_table.png` | Table | PNG/RGB | 900×440 | 22.3 KB | Visual IOC matrix | ❌ Phase 3 |

---

## Sample / Test Data (Phase 1 Baseline)

| File | Evidence Type | Modality | Size | Records | Notes |
|---|---|---|---|---|---|
| `sample_sysmon_logs.json` | Synthetic Sysmon events | JSON | 8.1 KB | 15 events | Labeled SYNTHETIC; used for Phase 1 tests |
| `sample_incident_report.txt` | Synthetic incident report | Text | 6.1 KB | 1 document | Labeled SYNTHETIC; used for Phase 1 tests |
| `sample_firewall_logs.csv` | Synthetic firewall logs | CSV | 1.3 KB | 10 rows | Labeled SYNTHETIC; used for Phase 1 tests |

---

## MITRE ATT&CK STIX Object Breakdown

| Object Type | Count |
|---|---|
| `relationship` | 21,262 |
| `x-mitre-analytic` | 1,758 |
| `attack-pattern` (techniques) | 858 |
| `malware` | 733 |
| `x-mitre-detection-strategy` | 699 |
| `course-of-action` (mitigations) | 268 |
| `intrusion-set` (threat groups) | 191 |
| `x-mitre-data-component` | 109 |
| `tool` | 95 |
| `campaign` | 56 |
| `x-mitre-data-source` | 38 |
| `x-mitre-tactic` | 15 |
| Other | 4 |
| **Total** | **26,086** |

---

## Verified ATT&CK Technique IDs

The following technique IDs have been **programmatically confirmed** to exist in `enterprise-attack.json`:

| Technique ID | Confirmed |
|---|---|
| T1059 (Command and Scripting Interpreter) | ✅ YES |
| T1059.001 (PowerShell) | ✅ YES |
| T1003 (OS Credential Dumping) | ✅ YES |
| T1003.001 (LSASS Memory) | ✅ YES |
| T1087 (Account Discovery) | ✅ YES |
| T1190 (Exploit Public-Facing Application) | ✅ YES |
| T1546 (Event Triggered Execution) | ✅ YES |

---

## NVD CVE Records

| CVE ID | CVSS v3.1 Score | Severity | CWE | Description Summary |
|---|---|---|---|---|
| CVE-2021-44228 | 10.0 | CRITICAL | CWE-20 | Log4Shell: Apache Log4j2 JNDI injection RCE |
| CVE-2017-0144 | 8.8 | HIGH | NVD-CWE-noinfo | EternalBlue: SMBv1 Windows RCE |
| CVE-2021-34527 | 8.8 | HIGH | NVD-CWE-noinfo | PrintNightmare: Windows Print Spooler RCE |
| CVE-2021-26855 | 9.1 | CRITICAL | CWE-918 | ProxyLogon: Microsoft Exchange SSRF/RCE |
| CVE-2022-30190 | 7.8 | HIGH | NVD-CWE-noinfo | Follina: MSDT URL Protocol RCE |
| CVE-2023-34362 | 9.8 | CRITICAL | CWE-89 | MOVEit Transfer SQL Injection |
| CVE-2023-4966 | 9.4 | CRITICAL | CWE-119 | CitrixBleed: NetScaler information disclosure |

---

## Embedding / Indexing Status (as of 2026-09-07)

| Dataset | Raw ✅ | Processed ❌ | Embedded ❌ | In FAISS ❌ |
|---|---|---|---|---|
| Atomic-EVTX | ✅ | ❌ | ❌ | ❌ |
| Mordor | ✅ | ❌ | ❌ | ❌ |
| CIC-IDS2017 | ✅ | ❌ | ❌ | ❌ |
| MITRE ATT&CK | ✅ | ❌ | ❌ | ❌ |
| NVD CVE | ✅ | ❌ | ❌ | ❌ |
| PDF Documents | ✅ | ❌ | ❌ | ❌ |
| DOCX Documents | ✅ | ❌ | ❌ | ❌ |
| Images | ✅ | ❌ | ❌ | ❌ |
| **Sample data (Phase 1)** | ✅ | ✅ | ✅ | ✅ |

All 8 new research datasets are **raw-only**. They have been verified parseable by the Phase 1.5 loaders, but have **not yet been embedded or indexed**. This is correct and expected at the Phase 1.5 milestone.

---

## Data Provenance Chain (Verified)

```
EVTX:
  evidence_id: EVTX-Exec_sysmon_meterpreter_reversetcp_msipackage-0001
  source_file: Exec_sysmon_meterpreter_reversetcp_msipackage.evtx
  raw_reference: data\raw\atomic_evtx\Exec_sysmon_meterpreter_reversetcp_msipackage.evtx  [EXISTS]
  timestamp: 2019-04-30 22:48:58 UTC
  event_type: EventID_2
  attck_technique: T1059 (supplied at load time)

NETWORK FLOW:
  evidence_id: NET-cicids2017_representative_sample-00001
  source_file: cicids2017_representative_sample.csv
  raw_reference: data\raw\cic_ids2017\cicids2017_representative_sample.csv  [EXISTS]
  row_index: 0 (in metadata)

DOCX:
  5 records produced from soc_incident_investigation_briefing.docx
  4 section records + 1 table record
  raw_reference: data\raw\documents\docx\soc_incident_investigation_briefing.docx  [EXISTS]

IMAGE:
  evidence_id: IMG-siem_alert_powershell_execution
  raw_reference: data\raw\images\screenshots\siem_alert_powershell_execution.png  [EXISTS]
  content: METADATA ONLY (no OCR/VLM — Phase 3 pending)
```
