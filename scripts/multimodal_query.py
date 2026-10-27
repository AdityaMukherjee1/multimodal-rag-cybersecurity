#!/usr/bin/env python3
"""Execute multimodal cybersecurity investigation query across text and visual evidence.

Performs joint retrieval across:
1. Text & Structured Knowledge Store (1,551 BGE vectors: EVTX, Mordor, MITRE, NVD, CIC-IDS, PDF, DOCX)
2. Visual Evidence Store (4 CLIP vectors: Screenshots, Attack Diagrams, Visual IOC Tables)

Applies Reciprocal Rank Fusion / Weighted Score Normalization to rank candidates,
displaying full provenance and maintaining strict separation between Observed Evidence
and Model Interpretation.

Usage:
    python scripts/multimodal_query.py "What evidence indicates PowerShell credential access or LSASS dumping?"
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
from src.llm.llm_client import create_llm
from src.multimodal.multimodal_retriever import FusedRetrievalResult, MultimodalRetriever
from src.multimodal.visual_embedding_service import CLIPVisualEmbeddingService
from src.multimodal.visual_vector_store import VisualVectorStore
from src.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-32s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MultimodalQuery")


def main() -> None:
    """Execute multimodal investigation query."""
    parser = argparse.ArgumentParser(description="Multimodal investigation query over text and visual evidence.")
    parser.add_argument("query", type=str, help="Investigation question or query string.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of fused evidence results to return.")
    parser.add_argument(
        "--text-index-dir",
        type=str,
        default="data/processed/faiss_index",
        help="Path to text FAISS index directory.",
    )
    parser.add_argument(
        "--visual-index-dir",
        type=str,
        default="data/processed/visual_faiss_index",
        help="Path to visual FAISS index directory.",
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        default="weighted",
        choices=["weighted", "rrf"],
        help="Candidate ranking fusion method ('weighted' or 'rrf').",
    )
    args = parser.parse_args()

    settings = Settings()
    text_dir = PROJECT_ROOT / args.text_index_dir
    vis_dir = PROJECT_ROOT / args.visual_index_dir

    if not text_dir.exists():
        logger.error("Text FAISS index not found at %s. Run scripts/build_research_index.py first.", text_dir)
        sys.exit(1)

    start_time = time.time()

    # Load stores
    text_store = VectorStore.load(text_dir, dimension=settings.embedding_dimension)
    visual_store = None
    if vis_dir.exists():
        try:
            visual_store = VisualVectorStore.load(vis_dir, dimension=512)
        except Exception as e:
            logger.warning("Could not load visual index: %s", e)

    text_embed_service = EmbeddingService(model_name=settings.embedding_model_name)
    visual_embed_service = CLIPVisualEmbeddingService()

    retriever = MultimodalRetriever(
        text_store=text_store,
        visual_store=visual_store,
        text_embedding_service=text_embed_service,
        visual_embedding_service=visual_embed_service,
    )

    fused_results: list[FusedRetrievalResult] = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        fusion_method=args.fusion_method,
    )

    elapsed_ms = (time.time() - start_time) * 1000

    # Format output
    print("\n" + "=" * 75)
    print("MULTIMODAL INVESTIGATION QUERY RESULTS")
    print("=" * 75)
    print(f"Query: \"{args.query}\"")
    print(f"Retrieved {len(fused_results)} fused evidence item(s) in {elapsed_ms:.1f}ms\n")

    for res in fused_results:
        print("-" * 75)
        print(f"--- Evidence Rank #{res.rank} | Fused Score: {res.fused_score:.4f} ---")
        print(f"  Evidence ID:     {res.evidence_id}")
        print(f"  Source File:     {res.source_file} ({res.source_dataset})")
        print(f"  Modality:        {res.modality} (Retrieved via: {res.retrieval_modality})")
        print(f"  Evidence Nature: {res.evidence_nature}")
        print(f"  Raw Reference:   {res.raw_reference}")

        attck = res.metadata.get("attck_technique")
        cve = res.metadata.get("cve_id")
        if attck:
            print(f"  MITRE ATT&CK:    {attck}")
        if cve:
            print(f"  CVE ID:          {cve}")

        print("\n  Evidence Content:")
        content_lines = res.content.splitlines()
        for line in content_lines[:15]:
            print(f"    {line}")
        if len(content_lines) > 15:
            print(f"    ... [{len(content_lines) - 15} lines truncated] ...")
        print()

    print("=" * 75)
    print("EVIDENCE PROVENANCE AUDIT SUMMARY")
    print("=" * 75)
    for res in fused_results:
        exists = Path(res.raw_reference).exists() or (PROJECT_ROOT / res.raw_reference).exists()
        print(
            f"  [{res.rank}] ID: {res.evidence_id:<36s} | Modality: {res.modality:<16s} | "
            f"Disk File: {'EXISTS' if exists else 'MISSING'}"
        )
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
