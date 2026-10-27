#!/usr/bin/env python3
"""Query the RAG pipeline with a cybersecurity investigation question.

Usage:
    python scripts/query.py "Your question here"
    python scripts/query.py "Was suspicious PowerShell activity detected?" --top-k 5

Prerequisite:
    Run scripts/build_index.py first to build the FAISS index.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Settings
from src.pipeline.rag_pipeline import RAGPipeline


def main() -> None:
    """Query the pipeline with a cybersecurity question."""
    parser = argparse.ArgumentParser(
        description="Query the RAG pipeline for cybersecurity investigation."
    )
    parser.add_argument(
        "question",
        type=str,
        help="Cybersecurity investigation question",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of evidence chunks to retrieve (default: from config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level (default: from config)",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save full results as JSON (for research logging)",
    )
    args = parser.parse_args()

    # Load settings
    settings = Settings()
    if args.log_level:
        settings.log_level = args.log_level

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("CyberAI-Investigator — Query Pipeline")
    logger.info("=" * 60)

    # Load index and query
    pipeline = RAGPipeline(settings)
    pipeline.load_index()

    result = pipeline.query(
        question=args.question,
        top_k=args.top_k,
    )

    # Print the answer
    print("\n" + result["answer"])

    # Print evidence summary
    print("\n" + "=" * 60)
    print("QUERY METADATA")
    print("=" * 60)
    metadata = result["query_metadata"]
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # Print evidence sources
    print("\n" + "=" * 60)
    print("EVIDENCE SOURCES")
    print("=" * 60)
    for ev in result["evidence"]:
        print(f"  [{ev['rank']}] {ev['source']} (chunk {ev['chunk_index']}, score: {ev['score']:.4f})")

    # Save results if requested
    if args.save_results:
        save_path = Path(args.save_results)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
