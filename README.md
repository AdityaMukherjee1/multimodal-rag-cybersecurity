# CyberAI-Investigator

**Evidence-Grounded Multimodal RAG for Cybersecurity Incident Investigation**

> A research-oriented AI cybersecurity investigation assistant that uses Retrieval-Augmented Generation (RAG) to retrieve relevant evidence and generate grounded incident investigation responses.

---

## Project Objective

Develop an AI-powered cybersecurity investigation assistant that:

1. Accepts heterogeneous cybersecurity evidence (logs, reports, alerts, screenshots)
2. Uses RAG to retrieve relevant evidence from a vector database
3. Generates incident investigation responses grounded in the retrieved evidence
4. Verifies whether generated claims are actually supported by the evidence
5. Reduces hallucination in cybersecurity analysis

## Research Problem

Large Language Models (LLMs) can generate plausible-sounding cybersecurity analysis that is not grounded in actual evidence. This project investigates:

- How RAG improves the accuracy and grounding of cybersecurity incident analysis
- How multimodal evidence (text + images) affects investigation quality
- How evidence verification mechanisms can detect and reduce hallucination
- Comparative evaluation of different RAG architectures for cybersecurity

## Current Development Phase

### ✅ Phase 1 — Basic Text RAG (Current)

The foundational text-based RAG pipeline:

- Document ingestion (`.txt`, `.log`, `.json`, `.csv`, `.pdf`)
- Text preprocessing and chunking
- Embedding generation using SentenceTransformers
- FAISS vector storage and similarity search
- Modular LLM interface with placeholder for offline testing
- End-to-end pipeline orchestration
- Configurable parameters for research reproducibility

### 🔲 Phase 2 — Cybersecurity Knowledge *(Planned)*

- MITRE ATT&CK framework integration
- CVE database enrichment
- Threat intelligence feeds
- Security incident report corpus

### 🔲 Phase 3 — Multimodal Capability *(Planned)*

- Screenshot and image processing
- Table and chart extraction
- Visual security evidence analysis
- Multimodal embedding and retrieval

### 🔲 Phase 4 — Evidence Verification *(Planned)*

- Claim extraction from LLM responses
- Evidence-claim matching
- Hallucination detection
- Confidence scoring
- Re-retrieval for unsupported claims

### 🔲 Phase 5 — Evaluation *(Planned)*

- Baseline: LLM without RAG
- Standard text RAG
- Multimodal RAG
- Multimodal RAG + Evidence Verification
- Metrics: Accuracy, Precision, Recall, F1, Hallucination Rate, Evidence Support Rate, Retrieval Recall@K

## Architecture

```
Cybersecurity Evidence (logs, reports, alerts)
         │
         ▼
┌─────────────────┐
│  Data Ingestion  │  ← DocumentLoader
└────────┬────────┘
         ▼
┌─────────────────┐
│  Preprocessing   │  ← TextProcessor (clean + chunk)
└────────┬────────┘
         ▼
┌─────────────────┐
│   Embeddings     │  ← EmbeddingService (SentenceTransformers)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Vector Store    │  ← VectorStore (FAISS)
└────────┬────────┘
         ▼
┌─────────────────┐
│   Retrieval      │  ← Similarity search (top-K)
└────────┬────────┘
         ▼
┌─────────────────┐
│   LLM Client     │  ← BaseLLM interface (pluggable providers)
└────────┬────────┘
         ▼
┌─────────────────┐
│   Response       │  ← Answer + Retrieved Evidence + Metadata
└─────────────────┘
```

**[PLANNED]** Future phases will add:
- Multimodal processing between Ingestion and Embeddings
- Evidence Verification after LLM response generation
- MITRE ATT&CK knowledge enrichment

## Technology Stack

| Component | Technology | Status |
|---|---|---|
| Language | Python 3.11+ | ✅ Active |
| Embeddings | SentenceTransformers (BGE-small-en-v1.5) | ✅ Active |
| Vector DB | FAISS | ✅ Active |
| PDF Processing | PyMuPDF | ✅ Active |
| Data Processing | Pandas | ✅ Active |
| Configuration | python-dotenv + PyYAML | ✅ Active |
| LLM | Modular interface (placeholder) | ✅ Interface Ready |
| Backend | FastAPI | 🔲 Planned |
| Frontend | Streamlit | 🔲 Planned |
| Multimodal | Vision model integration | 🔲 Planned |
| Knowledge | MITRE ATT&CK, CVE | 🔲 Planned |

## Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package installer)
- Git

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd CyberAI-Investigator

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
copy .env.example .env
```

### Configuration

Edit `.env` or `config.yaml` to configure:

- **Embedding model**: Change `EMBEDDING_MODEL_NAME` to use a different model
- **Chunk size**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` for experimentation
- **Top-K retrieval**: Set `TOP_K` for the number of evidence chunks to retrieve
- **LLM provider**: Set `LLM_PROVIDER` when you have an API key available

## How to Run — Phase 1 Prototype

### Step 1: Build the FAISS Index

```bash
python scripts/build_index.py
```

This loads sample cybersecurity data, processes it into chunks, generates embeddings, and stores them in a FAISS index.

### Step 2: Query the Pipeline

```bash
python scripts/query.py "Was suspicious PowerShell activity detected?"
```

This retrieves the most relevant evidence chunks and generates a response.

### Additional Options

```bash
# Custom data directory
python scripts/build_index.py --data-dir data/your_data

# Custom top-K
python scripts/query.py "What network connections were detected?" --top-k 10

# Save results for research analysis
python scripts/query.py "Describe the incident timeline" --save-results results/experiment1.json

# Verbose logging
python scripts/query.py "Your question" --log-level DEBUG
```

### Research Data Layer & Inventory

Inspect the multimodal research datasets:

```bash
# Display full research data inventory
python scripts/inventory_datasets.py

# Re-acquire or refresh authentic research datasets
python scripts/acquire_research_data.py
```

### Running Tests

```bash
python -m pytest tests/ -v
```

## Project Directory Structure

```
CyberAI-Investigator/
│
├── data/
│   ├── raw/                    # User's real datasets (gitignored)
│   ├── processed/              # Generated FAISS indices and processed data
│   └── sample/                 # Synthetic sample data for testing
│       ├── sample_sysmon_logs.json
│       ├── sample_incident_report.txt
│       └── sample_firewall_logs.csv
│
├── src/
│   ├── config/
│   │   └── settings.py         # Centralized configuration
│   ├── ingestion/
│   │   └── document_loader.py  # File loading (txt, json, csv, pdf)
│   ├── preprocessing/
│   │   └── text_processor.py   # Text cleaning and chunking
│   ├── embeddings/
│   │   └── embedding_service.py # SentenceTransformer embeddings
│   ├── retrieval/
│   │   └── vector_store.py     # FAISS vector store
│   ├── llm/
│   │   ├── base_llm.py         # Abstract LLM interface
│   │   ├── placeholder_llm.py  # Offline testing placeholder
│   │   └── llm_client.py       # Provider factory
│   ├── pipeline/
│   │   └── rag_pipeline.py     # End-to-end pipeline orchestration
│   ├── multimodal/             # [PLANNED] Phase 3
│   ├── verification/           # [PLANNED] Phase 4
│   └── knowledge/              # [PLANNED] Phase 2
│
├── evaluation/                 # [PLANNED] Phase 5
├── notebooks/                  # Jupyter notebooks for exploration
├── tests/
│   ├── test_preprocessing.py   # TextProcessor unit tests
│   └── test_pipeline.py        # Pipeline smoke tests
├── scripts/
│   ├── build_index.py          # CLI: build FAISS index
│   └── query.py                # CLI: query the pipeline
├── frontend/                   # [PLANNED] Streamlit UI
├── docs/                       # Documentation
│
├── config.yaml                 # Default configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Future Phases

> **Note:** The following features are planned but NOT yet implemented.

### Phase 2: Cybersecurity Knowledge Integration
- MITRE ATT&CK technique mapping
- CVE database enrichment for vulnerability context
- Threat intelligence feed integration
- Domain-specific knowledge graphs

### Phase 3: Multimodal Evidence Processing
- Screenshot analysis using vision models
- Table and chart extraction from images
- Network diagram understanding
- Multimodal embedding and cross-modal retrieval

### Phase 4: Evidence Verification & Hallucination Detection
- Automatic claim extraction from LLM responses
- Claim-evidence matching and verification
- Confidence scoring for each generated claim
- Re-retrieval and correction for unsupported claims

### Phase 5: Comparative Evaluation
- Benchmarking across architectures (No-RAG, Text-RAG, Multimodal-RAG, RAG+Verification)
- Standard metrics: Accuracy, Precision, Recall, F1-score
- Custom metrics: Hallucination Rate, Evidence Support Rate
- Retrieval quality: Recall@K
- Latency and efficiency analysis

## License

This project is developed as part of a 4th-year undergraduate research project.

## Disclaimer

All sample data included in this project is **entirely synthetic** and was created for testing purposes. No real systems were compromised, no real attacks were performed, and all IP addresses and domain names use reserved/documentation ranges (RFC 5737, RFC 2606).
