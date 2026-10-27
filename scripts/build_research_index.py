#!/usr/bin/env python3
"""Build unified FAISS research index from multimodal cybersecurity evidence.

Ingests and vectorizes:
1. MITRE ATT&CK Enterprise STIX Knowledge Base
2. NIST NVD / CVE Vulnerability Feed
3. Atomic-EVTX & DFIR Windows Event Telemetry
4. OTRF Mordor Adversary Simulation Event Logs
5. CIC-IDS2017 Network Intrusion Flow Records
6. Incident Handling PDFs (NIST SP 800-61r2, CISA Advisories)
7. SOC Incident Investigation DOCX Briefings

Usage:
    python scripts/build_research_index.py [--output-dir data/processed/faiss_index]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings
from src.embeddings.embedding_service import EmbeddingService
from src.ingestion.docx_loader import DOCXLoader
from src.ingestion.evidence_schema import CanonicalEvidenceRecord
from src.ingestion.evtx_parser import EVTXParser
from src.ingestion.mordor_loader import MordorLoader
from src.ingestion.network_loader import NetworkFlowLoader
from src.ingestion.pdf_loader import PDFLoader
from src.knowledge.attack_loader import AttackKnowledgeBase
from src.knowledge.cve_indexer import CVEKnowledgeBase
from src.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-32s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BuildResearchIndex")


def build_knowledge_indices(data_dir: Path) -> tuple[list[CanonicalEvidenceRecord], list[CanonicalEvidenceRecord]]:
    """Build and load MITRE ATT&CK and NVD CVE knowledge indices."""
    knowledge_processed_dir = data_dir / "processed" / "knowledge"
    knowledge_processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. MITRE ATT&CK STIX
    attack_stix_file = data_dir / "raw" / "mitre_attack" / "enterprise-attack.json"
    attack_index_file = knowledge_processed_dir / "attack_index.json"
    attack_kb = AttackKnowledgeBase()

    if attack_stix_file.exists():
        logger.info("Building MITRE ATT&CK knowledge index...")
        attack_kb.build_from_stix(attack_stix_file, attack_index_file)
        attack_records = attack_kb.to_canonical_records(str(attack_stix_file))
    elif attack_index_file.exists():
        logger.info("Loading existing MITRE ATT&CK index...")
        attack_kb.load_index(attack_index_file)
        attack_records = attack_kb.to_canonical_records()
    else:
        logger.warning("MITRE ATT&CK file not found at %s", attack_stix_file)
        attack_records = []

    # 2. NVD CVEs
    nvd_file = data_dir / "raw" / "nvd" / "nvd_cve_sample.json"
    cve_index_file = knowledge_processed_dir / "cve_index.json"
    cve_kb = CVEKnowledgeBase()

    if nvd_file.exists():
        logger.info("Building NVD CVE knowledge index...")
        cve_kb.build_from_nvd(nvd_file, cve_index_file)
        cve_records = cve_kb.to_canonical_records(str(nvd_file))
    elif cve_index_file.exists():
        logger.info("Loading existing NVD CVE index...")
        cve_kb.load_index(cve_index_file)
        cve_records = cve_kb.to_canonical_records()
    else:
        logger.warning("NVD file not found at %s", nvd_file)
        cve_records = []

    return attack_records, cve_records


def ingest_evtx_records(data_dir: Path) -> list[CanonicalEvidenceRecord]:
    """Ingest all binary EVTX files."""
    evtx_dir = data_dir / "raw" / "atomic_evtx"
    if not evtx_dir.exists():
        return []

    technique_hints = {
        "Exec_sysmon_meterpreter_reversetcp_msipackage": "T1059",
        "4799_remote_local_groups_enumeration": "T1087.002",
        "Persistence_Winsock_Catalog_Change_EventId_1": "T1546",
        "CA_DCSync_4662": "T1003.006",
        "T1003-1_Sysmon_CredentialDumping": "T1003.001",
    }

    parser = EVTXParser()
    all_evtx_records: list[CanonicalEvidenceRecord] = []

    for evtx_path in sorted(evtx_dir.glob("*.evtx")):
        tech = technique_hints.get(evtx_path.stem)
        try:
            records = parser.parse_file(evtx_path, source_dataset="atomic_evtx", attck_technique=tech)
            all_evtx_records.extend(records)
        except Exception as e:
            logger.error("Failed to parse EVTX %s: %s", evtx_path.name, e)

    return all_evtx_records


def ingest_mordor_records(data_dir: Path, max_per_dataset: int = 150) -> list[CanonicalEvidenceRecord]:
    """Ingest Mordor simulated adversary NDJSON event logs."""
    mordor_dir = data_dir / "raw" / "mordor"
    if not mordor_dir.exists():
        return []

    loader = MordorLoader(max_records=max_per_dataset)
    all_mordor_records: list[CanonicalEvidenceRecord] = []

    mordor_configs = [
        ("psh_powershell_httplistener_*.json", "T1059.001", "Execution"),
        ("cmd_psexec_lsa_secrets_dump_*.json", "T1003", "Credential Access"),
        ("cmd_seatbelt_group_user_*.json", "T1087", "Discovery"),
    ]

    for pattern, tech, tactic in mordor_configs:
        for json_path in sorted(mordor_dir.glob(pattern)):
            try:
                recs = loader.load_file(json_path, source_dataset="mordor", attck_technique=tech, tactic=tactic)
                all_mordor_records.extend(recs)
            except Exception as e:
                logger.error("Failed to parse Mordor dataset %s: %s", json_path.name, e)

    return all_mordor_records


def ingest_network_records(data_dir: Path, max_records: int = 300) -> list[CanonicalEvidenceRecord]:
    """Ingest representative network flow records."""
    cic_csv = data_dir / "raw" / "cic_ids2017" / "cicids2017_representative_sample.csv"
    if not cic_csv.exists():
        return []

    loader = NetworkFlowLoader(max_records=max_records)
    try:
        return loader.load_file(cic_csv, source_dataset="cic_ids2017")
    except Exception as e:
        logger.error("Failed to load CIC-IDS2017: %s", e)
        return []


def ingest_documents(data_dir: Path) -> list[CanonicalEvidenceRecord]:
    """Ingest PDF and DOCX documents with section and page level provenance."""
    all_doc_records: list[CanonicalEvidenceRecord] = []

    # 1. PDFs
    pdf_dir = data_dir / "raw" / "documents" / "pdf"
    if pdf_dir.exists():
        pdf_loader = PDFLoader(skip_withdrawn_notices=True)
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            try:
                recs = pdf_loader.load_file(pdf_path, source_dataset="documents_pdf")
                all_doc_records.extend(recs)
            except Exception as e:
                logger.error("Failed to load PDF %s: %s", pdf_path.name, e)

    # 2. DOCX
    docx_dir = data_dir / "raw" / "documents" / "docx"
    if docx_dir.exists():
        docx_loader = DOCXLoader()
        for docx_path in sorted(docx_dir.glob("*.docx")):
            try:
                recs = docx_loader.load_file(docx_path, source_dataset="documents_docx")
                all_doc_records.extend(recs)
            except Exception as e:
                logger.error("Failed to load DOCX %s: %s", docx_path.name, e)

    return all_doc_records


def main() -> None:
    """Execute unified research data indexing."""
    parser = argparse.ArgumentParser(description="Build unified FAISS research index from multimodal evidence.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/faiss_index",
        help="Path to save the generated FAISS index and document metadata.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    args = parser.parse_args()

    start_time = time.time()
    data_dir = PROJECT_ROOT / "data"
    output_path = PROJECT_ROOT / args.output_dir

    logger.info("=" * 70)
    logger.info("STARTING PHASE 2 RESEARCH DATA INGESTION & VECTOR INDEXING")
    logger.info("=" * 70)

    # 1. Knowledge Base Ingestion (MITRE ATT&CK & NVD)
    attack_records, cve_records = build_knowledge_indices(data_dir)

    # 2. Host Telemetry Ingestion (EVTX & Mordor)
    evtx_records = ingest_evtx_records(data_dir)
    mordor_records = ingest_mordor_records(data_dir, max_per_dataset=150)

    # 3. Network Flow Ingestion (CIC-IDS2017)
    network_records = ingest_network_records(data_dir, max_records=300)

    # 4. Document Ingestion (PDF & DOCX)
    doc_records = ingest_documents(data_dir)

    # Combine all evidence records
    all_evidence: list[CanonicalEvidenceRecord] = (
        attack_records
        + cve_records
        + evtx_records
        + mordor_records
        + network_records
        + doc_records
    )

    logger.info("=" * 70)
    logger.info("COLLECTED EVIDENCE SUMMARY:")
    logger.info("  - MITRE ATT&CK Techniques: %d", len(attack_records))
    logger.info("  - NVD Vulnerabilities:     %d", len(cve_records))
    logger.info("  - Atomic-EVTX Events:      %d", len(evtx_records))
    logger.info("  - Mordor Adversary Events: %d", len(mordor_records))
    logger.info("  - CIC-IDS2017 Flow Rows:   %d", len(network_records))
    logger.info("  - Document Pages/Sections: %d", len(doc_records))
    logger.info("  --------------------------------------")
    logger.info("  TOTAL EVIDENCE UNITS:      %d", len(all_evidence))
    logger.info("=" * 70)

    if not all_evidence:
        logger.error("No evidence records collected. Aborting index build.")
        sys.exit(1)

    # Convert to VectorStore-compatible document dictionaries
    documents = [rec.to_document_dict() for rec in all_evidence]
    texts = [doc["content"] for doc in documents]

    # Initialize Embedding Service (BGE-small baseline)
    settings = Settings()
    embedding_service = EmbeddingService(
        model_name=settings.embedding_model_name,
    )

    logger.info("Generating embeddings for %d evidence units (batch_size=%d)...", len(texts), args.batch_size)
    embeddings = embedding_service.embed_texts(texts, batch_size=args.batch_size)

    # Build and Save FAISS Index
    vector_store = VectorStore(dimension=embedding_service.dimension)
    vector_store.add_documents(documents=documents, embeddings=embeddings)
    vector_store.save(output_path)

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("PHASE 2 RESEARCH INDEX BUILD COMPLETE in %.2f seconds", elapsed)
    logger.info("  - Total Vectors Indexed: %d", vector_store._index.ntotal)
    logger.info("  - Embedding Dimension:   %d", embedding_service.dimension)
    logger.info("  - Embedding Model:       %s", settings.embedding_model_name)
    logger.info("  - FAISS Index Path:      %s", output_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
