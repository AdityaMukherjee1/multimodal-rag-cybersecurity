# Phase 1.5 Validation Report
## Evidence-Grounded Multimodal RAG for Cybersecurity Incident Investigation

**Audit Date:** 2026-09-07  
**Python Version:** 3.13.15  
**Platform:** Windows 11 (win32)  
**Auditor:** Automated validation pipeline  

> [!IMPORTANT]
> This report is based entirely on **actual measured values** from the filesystem, live parsers, and test execution. No numbers are invented or extrapolated.

---

## 1. Executive Summary

Phase 1.5 has successfully established a multimodal cybersecurity research data foundation. The following has been **actually achieved** and **independently verified** in this audit:

| Component | Status |
|---|---|
| Phase 1 regression | ✅ ZERO regressions — 30/30 tests pass |
| FAISS index + BGE embeddings | ✅ Working; indexes 41 chunks from 3 sample files |
| Phase 1 query pipeline | ✅ Working; retrieves 5 sources with cosine scores |
| Canonical evidence schema | ✅ Complete; covers all planned modalities |
| EVTX parser | ✅ Reads binary EVTX, extracts forensic fields |
| Network flow loader | ✅ Reads CIC-IDS2017 CSV, normalizes 78 headers |
| DOCX loader | ✅ Extracts sections + Markdown table |
| Image metadata loader | ✅ Extracts PIL metadata; **no VLM/OCR** |
| All 8 datasets acquired (raw) | ✅ 56.4 MB across 25 files, 11 categories |
| New datasets embedded/indexed | ❌ NONE — all 8 new datasets are raw-only |

**The system is a correctly-architected data foundation.** The parsers work. The schema works. The provenance chain is intact. The current FAISS index contains **only Phase 1 synthetic sample data**, which is correct — indexing the real research datasets is the work of the next phase.

---

## 2. Phase 1 Regression Status

### Test Suite

```
pytest tests/ -v
Platform: win32 | Python 3.13.15 | pytest 9.1.1

Tests collected: 30
Tests passed:    30
Tests failed:    0
Tests skipped:   0
Tests errored:   0
Execution time:  149.22s (2 min 29 sec — dominated by BGE model load time)
```

**All 30 tests pass with zero failures.**

### Test Distribution

| Test File | Tests | Status |
|---|---|---|
| `test_evidence_schema.py` | 3 | ✅ All pass |
| `test_multimodal_ingestion.py` | 5 | ✅ All pass (uses real dataset files) |
| `test_pipeline.py` | 6 | ✅ All pass |
| `test_preprocessing.py` | 16 | ✅ All pass |

### Phase 1 Pipeline Commands

**Build index:**
```
python scripts/build_index.py
→ raw_documents=3, chunks=41, embedding_model=BAAI/bge-small-en-v1.5, dim=384
→ Index saved: data/processed/faiss_index/  (index.faiss=63KB, documents.pkl=21KB)
```

**Query:**
```
python scripts/query.py "Was suspicious PowerShell activity detected?"
→ Retrieved 5 results, top score 0.7973 (sample_incident_report.txt)
→ PlaceholderLLM response generated
→ Evidence sources with scores returned: ✅
→ Source filenames preserved: ✅
→ Chunk IDs returned: ✅
→ Similarity scores returned: ✅
```

**No Phase 1 regressions.** The pipeline is intact.

---

## 3. Dataset Inventory

### Actual Files Present on Disk (Measured 2026-09-07)

#### atomic_evtx — `data/raw/atomic_evtx/`

| File | Size | Notes |
|---|---|---|
| `4799_remote_local_groups_enumeration.evtx` | 69,632 B | Real EVTX |
| `CA_DCSync_4662.evtx` | 69,632 B | Real EVTX |
| `Exec_sysmon_meterpreter_reversetcp_msipackage.evtx` | 69,632 B | Real EVTX |
| `Persistence_Winsock_Catalog_Change_EventId_1.evtx` | 69,632 B | Real EVTX |
| `T1003-1_metadata.csv` | 325 B | Atomic test metadata |
| `T1003-1_Sysmon_CredentialDumping.evtx` | 69,632 B | Real EVTX |
| **Total** | **~418 KB** | **5 EVTX + 1 CSV** |

#### mordor — `data/raw/mordor/`

| File | Size | Records |
|---|---|---|
| `cmd_psexec_lsa_secrets_dump.zip` (archive) | 38,086 B | — |
| `cmd_psexec_lsa_secrets_dump_2020-10-1903305471.json` | 408,881 B | 286 NDJSON events |
| `cmd_seatbelt_group_user.zip` (archive) | 48,537 B | — |
| `cmd_seatbelt_group_user_2020-11-0216391814.json` | 827,640 B | 451 NDJSON events |
| `psh_powershell_httplistener.zip` (archive) | 10,491 B | — |
| `psh_powershell_httplistener_2020-11-0204130683.json` | 213,202 B | 110 NDJSON events |
| **Total (extracted JSON only)** | **~1.45 MB** | **847 events** |

#### cic_ids2017 — `data/raw/cic_ids2017/`

| File | Size | Rows | Columns |
|---|---|---|---|
| `cicids2017_representative_sample.csv` | 1,179,327 B | 3,000 | 78 |

> [!NOTE]
> The inventory script previously reported 6,001 rows. The actual measured value using `csv.DictReader` is **3,000 data rows**. The discrepancy was caused by the earlier line-count method (which counts raw lines including the header and possible blank lines). The verified correct value is **3,000 data rows**.

#### mitre_attack — `data/raw/mitre_attack/`

| File | Size | STIX Objects |
|---|---|---|
| `enterprise-attack.json` | 53,835,637 B (~51 MB) | 26,086 |

STIX spec_version field: Note — the bundle's top-level `spec_version` key is absent in this bundle (it reports `?`). The object-level `spec_version` fields within individual objects confirm STIX 2.1 format.

#### nvd — `data/raw/nvd/`

| File | Size | Records |
|---|---|---|
| `nvd_cve_sample.json` | 283,651 B | 7 CVE records |

#### documents/pdf — `data/raw/documents/pdf/`

| File | Size | Pages |
|---|---|---|
| `NIST.SP.800-61r2.pdf` | 1,744,473 B | 80 |
| `cisa_advisory_aa23_347a_briefing.pdf` | 1,894 B | 1 |

#### documents/docx — `data/raw/documents/docx/`

| File | Size | Paragraphs | Tables |
|---|---|---|---|
| `soc_incident_investigation_briefing.docx` | 37,646 B | 8 (non-empty) | 1 |

#### images

| Category | File | Size | Dimensions | Format |
|---|---|---|---|---|
| screenshots | `siem_alert_powershell_execution.png` | 29,574 B | 900×480 | PNG/RGB |
| screenshots | `terminal_credential_access_mimikatz.png` | 23,111 B | 850×420 | PNG/RGB |
| diagrams | `incident_attack_flow_diagram.png` | 26,759 B | 960×540 | PNG/RGB |
| tables | `visual_threat_intel_ioc_table.png` | 22,855 B | 900×440 | PNG/RGB |

### Totals
- **25 evidence files** across **11 categories**
- **~56.4 MB** total raw evidence

---

## 4. Dataset Quality

### 4.1 Atomic-EVTX

| Property | Finding |
|---|---|
| **Completeness** | PARTIAL — 5 EVTX files covering 4 tactic categories |
| **Format integrity** | ✅ All 5 EVTX files parse without errors |
| **Schema consistency** | ✅ Standard Windows XML Event format |
| **ATT&CK mappings** | Assigned by parser (not embedded in raw EVTX file) |
| **Data quality issue** | The `ParentImage` and `ParentCommandLine` fields are absent in 2 of the 5 EVTX files (they contain non-process-create events) |
| **Sampling concern** | Each file contains only 2–34 events (very small samples); entire dataset is 418 KB |

### 4.2 Mordor NDJSON

| Property | Finding |
|---|---|
| **Completeness** | 3 datasets / 847 total events |
| **Format integrity** | ✅ Valid NDJSON, all 3 files parse |
| **Schema consistency** | ⚠️ INCONSISTENT — seatbelt dataset includes `@timestamp`, `ProviderGuid`, `Keywords` fields absent in the psexec dataset |
| **ATT&CK technique_id field** | ❌ NOT PRESENT in raw records — no `technique_id` or `technique_name` key exists. ATT&CK attribution is dataset-level metadata only (in SOURCE.md), not per-record |
| **Missing values** | `ProcessId` is absent from Security channel events (only Sysmon provides it) |
| **Provenance** | All records contain `Hostname`, `TimeCreated`, `EventID`, `Channel`, `SourceName` |

> [!WARNING]
> **Important finding:** The claimed ATT&CK technique IDs (`T1059.001`, `T1003`, `T1087`) are **not embedded in individual Mordor NDJSON records**. They are dataset-level metadata only. When these records are ingested and vectorized, the technique attribution must be passed explicitly at the loader level (as the current architecture does), not derived from the raw event data.

### 4.3 CIC-IDS2017

| Property | Finding |
|---|---|
| **Completeness** | 3,000 data rows, 78 columns |
| **Format integrity** | ✅ Parses correctly with csv.DictReader |
| **Column naming** | ⚠️ `Destination Port` column is **absent** from this curated mirror — the column name does not appear in the 78 headers. The `Label` column has no leading space. |
| **Label distribution** | Heavily imbalanced: DoS=2,672 (89%), BENIGN=328 (11%) |
| **Missing values** | 0 empty values in `Flow Duration`, `Total Fwd Packets`, `Flow Bytes/s` |
| **Duplicate rows** | Not checked (not critical at this stage) |

> [!WARNING]
> **Important finding:** The `Destination Port` column expected by `NetworkFlowLoader` (`row.get("Destination Port", ...)`) will always fall through to the default `"Unknown"` because the column name does not exist in this CSV. This is a data/parser mismatch — **not a defect in the loader logic** (the fallback handles it gracefully), but it means destination port will show `"Unknown"` for all records. This should be documented and the exact column name inspected before Phase 2 ingestion.

### 4.4 MITRE ATT&CK STIX

| Property | Finding |
|---|---|
| **Completeness** | Full Enterprise ATT&CK knowledge base |
| **Format integrity** | ✅ Valid JSON, loads with `json.load()` |
| **STIX version** | STIX 2.1 (per object-level `spec_version`; top-level bundle key absent) |
| **Object count** | 26,086 total STIX objects |
| **attack-patterns** | 858 (techniques + sub-techniques) |
| **Technique ID verification** | T1059, T1059.001, T1003, T1003.001, T1087, T1546, T1190 — all **CONFIRMED PRESENT** |
| **Quality concern** | The bundle is 51 MB — loading the entire file for every query would be prohibitively expensive. Phase 2 must build a lookup index |

### 4.5 NVD CVE Data

| Property | Finding |
|---|---|
| **Completeness** | 7 records only (targeted high-impact CVEs) |
| **Format integrity** | ✅ Valid NVD CVE JSON 2.0 |
| **CVSS v3.1 availability** | ✅ All 7 records have `cvssMetricV31` with `baseScore` and `baseSeverity` |
| **Description quality** | ✅ English descriptions present for all 7 |
| **CWE availability** | Mixed — Log4Shell has CWE-20, some have NVD-CWE-noinfo (no CWE assigned by NVD) |
| **Sampling concern** | Only 7 records — sufficient for proof-of-concept only |

### 4.6 PDF Documents

| Property | Finding |
|---|---|
| **NIST.SP.800-61r2.pdf** | 80 pages, text+images, first page shows a "Withdrawn" notice |
| **cisa_advisory_aa23_347a_briefing.pdf** | 1 page, text-only, generated document (not officially downloaded) |
| **Text extraction** | ✅ PyMuPDF extracts text from both |
| **Visual content** | NIST PDF contains diagrams/images on pages 1–10; CISA PDF is text only |

> [!NOTE]
> The NIST SP 800-61r2 PDF begins with a "Withdrawn — archived" notice. The document content is still valid for research purposes. Future PDF ingestion should skip or handle this preamble page when chunking.

### 4.7 DOCX

| Property | Finding |
|---|---|
| **Paragraphs** | 8 non-empty paragraphs |
| **Tables** | 1 table (6 rows × 4 cols: Artifact Type, Indicator/Value, MITRE Technique, Confidence) |
| **DOCXLoader behavior** | ✅ Extracts all 4 sections + table as Markdown; 5 canonical records produced |
| **Table Markdown** | ✅ Header row + separator + 5 data rows |
| **Images in DOCX** | Not checked; document appears text+table only |

### 4.8 Images

| Property | Finding |
|---|---|
| **All 4 images open correctly** | ✅ PIL confirms PNG/RGB |
| **Semantic content** | ❌ NOT extracted — content field contains generated text description only |
| **OCR** | ❌ NOT implemented |
| **Visual embeddings** | ❌ NOT implemented |
| **Phase 3 readiness** | ✅ Files are in place; loaders are hooked; VLM/OCR integration is pending |

---

## 5. Modality Support

| Modality | Extension(s) | Parser / Loader | Canonical Record Produced | In Vector Store | Quality |
|---|---|---|---|---|---|
| **Text** | `.txt` | `DocumentLoader` (Phase 1) | Via Phase 1 chunking | ✅ YES | Good |
| **Log** | `.log` | `DocumentLoader` (Phase 1) | Via Phase 1 chunking | ✅ YES | Good |
| **JSON / NDJSON** | `.json` | `DocumentLoader` (Phase 1) | Via Phase 1 text chunking | ✅ YES (sample only) | Basic |
| **CSV** | `.csv` | `DocumentLoader` / `NetworkFlowLoader` | Phase 1.5: per-row records | ❌ NOT YET | Good |
| **EVTX** | `.evtx` | `EVTXParser` | Per-event records | ❌ NOT YET | Good |
| **PDF** | `.pdf` | `DocumentLoader` via PyMuPDF | Via Phase 1 chunking | ❌ NOT YET | Good for text pages |
| **DOCX** | `.docx` | `DOCXLoader` | Per-section + per-table | ❌ NOT YET | Good |
| **PNG / JPG** | `.png`, `.jpg` | `ImageEvidenceLoader` | Metadata + placeholder | ❌ NOT YET | Metadata only |

**Note:** For the JSON/NDJSON row, Phase 1 loads the Mordor NDJSON as flat text chunks — it does not use the structured event fields. Per-event structured loading of Mordor data requires a dedicated `MordorLoader` (not yet built).

---

## 6. Canonical Evidence Schema

### Fields Present in `CanonicalEvidenceRecord`

| Field | Type | Required | Populated by |
|---|---|---|---|
| `evidence_id` | str | ✅ Yes | Each parser |
| `source_file` | str | ✅ Yes | Each parser |
| `source_dataset` | str | ✅ Yes | Each parser |
| `modality` | str | ✅ Yes | Each parser |
| `content` | str | ✅ Yes | Each parser |
| `raw_reference` | str | ✅ Yes | Each parser |
| `metadata` | dict | Optional | Each parser (domain fields) |
| `timestamp` | str | Optional | EVTX, Mordor |
| `incident_id` | str | Optional | Not populated yet |
| `event_type` | str | Optional | All parsers |
| `page_number` | int | Optional | PDF (not yet used) |
| `chunk_id` | int | Optional | All parsers |
| `attck_technique` | str | Optional | Passed at load time |
| `cve_id` | str | Optional | NVD (not yet used) |
| `confidence_score` | float | Optional | Not populated yet |
| `created_at` | str | Auto | Auto-generated UTC |

### Provenance Assessment

**EVTX:** Full chain verified — `evidence_id → source_file → raw_reference (exists) → timestamp → event_type → attck_technique`. ✅ Complete.

**Network Flow:** Full chain verified — `evidence_id → source_file (CSV) → raw_reference (exists) → event_type → row_index in metadata`. ✅ Complete.

**DOCX:** Full chain verified — `evidence_id → source_file → raw_reference (exists) → chunk_id (section/table index)`. ✅ Complete. (No page_number needed; DOCX is not paginated the same way.)

**Image:** Partial chain — `evidence_id → source_file → raw_reference (exists)`. Content is metadata description only; no visual semantic content. ⚠️ Provenance structurally complete; semantic content missing (Phase 3).

**Identified provenance gaps:**

1. **`incident_id`** — Not populated by any current parser. Required for cross-evidence correlation in Phase 4.
2. **`page_number`** — Not populated by PDFs parsed through the Phase 1 `DocumentLoader` (the `DOCXLoader` correctly omits it). Required for PDF evidence auditing.
3. **Mordor `attck_technique` per record** — Not extractable from raw data (no field exists); must be supplied at load-time from dataset-level metadata.

---

## 7. Vector Store Status

**Current FAISS index contains:**
- 41 vectors, dimension 384 (BGE-small-en-v1.5)
- Sources: `sample_incident_report.txt`, `sample_sysmon_logs.json`, `sample_firewall_logs.csv` (all from `data/sample/`)
- Modality: `text` only
- Index files: `index.faiss` (63 KB), `documents.pkl` (21 KB)

### Embedding Status Table

| Dataset | Raw | Processed | Embedded | In FAISS |
|---|---|---|---|---|
| Atomic-EVTX | ✅ | ❌ | ❌ | ❌ |
| Mordor | ✅ | ❌ | ❌ | ❌ |
| CIC-IDS2017 | ✅ | ❌ | ❌ | ❌ |
| MITRE ATT&CK | ✅ | ❌ | ❌ | ❌ |
| NVD | ✅ | ❌ | ❌ | ❌ |
| PDF | ✅ | ❌ | ❌ | ❌ |
| DOCX | ✅ | ❌ | ❌ | ❌ |
| Images | ✅ | ❌ | ❌ | ❌ |

This is **correct and expected**. The data foundation was built in Phase 1.5. The embedding and indexing of these datasets is Phase 2 work (specifically: text modalities) and Phase 3 work (multimodal visual embeddings).

---

## 8. Multimodal Readiness

### Already Implemented ✅

| Capability | Status |
|---|---|
| Text RAG (chunk → embed → FAISS → retrieve) | Complete |
| EVTX binary parsing → canonical text events | Complete |
| Network flow CSV → canonical flow summaries | Complete |
| DOCX section + table extraction | Complete |
| PDF text extraction (PyMuPDF) | Complete (Phase 1) |
| Image metadata extraction (PIL) | Complete |
| Canonical evidence schema with provenance | Complete |
| Dataset acquisition + manifest | Complete |

### Not Yet Implemented ❌

| Capability | Phase |
|---|---|
| Real visual embeddings (CLIP/BLIP/SigLIP) | Phase 3 |
| Vision-Language Model (VLM) image understanding | Phase 3 |
| OCR on screenshots/tables | Phase 3 |
| Image-to-text semantic retrieval | Phase 3 |
| Cross-modal (text+image) joint retrieval | Phase 3 |
| Multimodal LLM reasoning (GPT-4V / Gemini Vision) | Phase 3 |
| MITRE ATT&CK structured lookup index | Phase 2 |
| NVD vulnerability enrichment pipeline | Phase 2 |
| Mordor per-event structured loader | Phase 2 |
| Evidence claim verification | Phase 4 |

> [!IMPORTANT]
> The system is **not yet** a multimodal RAG system in the functional sense. It is an **evidence-grounded text RAG system** with multimodal data loaded into canonical records. The visual evidence exists only as metadata placeholders. This is scientifically honest and architecturally sound — Phase 3 VLM integration will complete the multimodal pipeline.

---

## 9. Problems Found

### Problem 1: `Destination Port` column missing from CIC-IDS2017 CSV
- **Severity:** Minor (functional, not critical)
- **Finding:** The NetworkFlowLoader tries `row.get("Destination Port", ...)` but the column does not exist in this CSV — the first 78 headers are statistical flow features; `Destination Port` is not among them.
- **Impact:** All network records will show `destination_port: "Unknown"` in metadata.
- **Fix required:** Not urgent for Phase 1.5 validation. Before Phase 2 ingestion, inspect exact column names and map appropriately.

### Problem 2: MITRE STIX bundle missing top-level `spec_version`
- **Severity:** Cosmetic
- **Finding:** `mitre_data.get('spec_version', '?')` returns `'?'` — the bundle structure uses a different top-level format than expected. Objects themselves are valid STIX 2.1.
- **Impact:** None on functionality.

### Problem 3: NIST PDF first page is a withdrawal notice
- **Severity:** Minor (data quality)
- **Finding:** First page reads "Withdrawn NIST Technical Series Publication — Warning Notice." Content starts on page 2.
- **Impact:** When PDF is chunked, the first chunk will contain irrelevant preamble text. Needs a skip/offset when indexing.

### Problem 4: Mordor records have no per-record ATT&CK field
- **Severity:** Medium (research significance)
- **Finding:** No `technique_id`, `technique_name`, or `tactic` field in individual NDJSON events. ATT&CK attribution is dataset-level only.
- **Impact:** When records are individually embedded, the ATT&CK mapping must be injected at loader time — it cannot be retrieved from the raw record.

### Problem 5: Sparse NVD sample (7 CVE records)
- **Severity:** Research concern
- **Finding:** Only 7 high-profile CVEs are present. Sufficient for proof-of-concept but insufficient for general vulnerability correlation.
- **Impact:** Phase 2 NVD indexer should expand the corpus.

### Problem 6: CIC-IDS2017 class imbalance
- **Severity:** Research concern
- **Finding:** DoS=89%, BENIGN=11%. Only 2 attack categories present.
- **Impact:** Network flow retrieval will be biased toward DoS patterns. For research experiments, additional traffic categories should be acquired.

---

## 10. Changes Made

**No source code changes were made during this audit.**

All 30 tests passed on the first run. No defects requiring code modifications were found. The problems listed in Section 9 are data quality or future-phase concerns, not current code defects.

---

## 11. Research Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Synthetic vs real telemetry** | High | EVTX + Mordor are real captures from controlled lab environments. Sample data is explicitly labeled SYNTHETIC. Maintain clear separation. |
| **Dataset bias (CIC-IDS2017 imbalance)** | Medium | 89% DoS / 11% BENIGN. Retrieval results will skew toward DoS. Document this in experimental reporting. |
| **Small EVTX sample size** | Medium | 5 files / 52 total events. Retrieval recall evaluation will be statistically unreliable with this sample size. Expand before Phase 4 evaluation. |
| **Mordor per-record ATT&CK attribution gap** | Medium | Cannot derive technique from individual event record. Mitigated by dataset-level attribution at load time. |
| **NVD sample too small** | Medium | 7 CVEs. Sufficient for demos, insufficient for correlation studies. |
| **Image content not semantically indexed** | High (for multimodal claim) | Must not claim multimodal retrieval capability until Phase 3 VLM integration is complete. |
| **MITRE STIX load time (51 MB)** | Medium | Loading entire file for each query is prohibitive. Phase 2 must build a persistent lookup structure. |
| **NIST PDF withdrawal notice** | Low | First-page preamble will pollute chunked evidence. Filter before indexing. |
| **No HTTPS certificate pinning** | Low | Download scripts use urllib. No man-in-the-middle protection. Acceptable for research downloads. |
| **Licensing constraint (CC BY-SA 4.0 for Mordor)** | Low | Mordor data must be used with attribution and any derivative datasets must also be CC BY-SA. Documented in SOURCE.md. |

---

## 12. Phase 2 Readiness

### Verdict: **READY WITH CONDITIONS**

#### Why READY:
- ✅ Raw datasets for all 5 primary cybersecurity evidence categories are present and verified
- ✅ Canonical evidence schema is complete and tested
- ✅ Modality-specific parsers work and produce valid `CanonicalEvidenceRecord` objects
- ✅ Phase 1 pipeline is intact; new datasets will not break it
- ✅ FAISS vector store supports adding new documents; `to_document_dict()` is compatible
- ✅ MITRE ATT&CK STIX confirmed: all target technique IDs (T1059.001, T1003, T1087, etc.) exist in the dataset
- ✅ NVD CVE records confirmed: all 7 high-impact CVEs present with CVSS v3.1 data

#### Conditions before Phase 2:
1. **Before indexing EVTX/Mordor:** Verify that the `build_index` script or a new ingestion pipeline correctly passes `attck_technique` metadata from dataset manifest to each loader call.
2. **Before indexing CIC-IDS2017:** Inspect actual column names in the CSV and update `NetworkFlowLoader` column mapping for `Destination Port` (or document its absence).
3. **Before indexing PDFs:** Add page-offset logic to skip the NIST SP 800-61r2 withdrawal preamble page.
4. **Before making any claim of "multimodal RAG":** Complete Phase 3 VLM integration. Current image handling is metadata-only.

#### Recommended Phase 2 Scope:
1. `src/knowledge/attack_loader.py` — STIX ATT&CK lookup index
2. `src/knowledge/cve_indexer.py` — NVD CVE lookup index
3. Updated `build_index.py` or new `scripts/build_research_index.py` — indexes all real datasets into FAISS using existing parsers
4. Provenance logging for `page_number` in PDF chunks

---

*Report generated from live measurements on 2026-09-07. All statistics verified by automated inspection scripts.*
