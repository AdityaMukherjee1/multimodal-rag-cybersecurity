#!/usr/bin/env python3
"""Build the FAISS index from cybersecurity evidence documents.

Usage:
    python scripts/build_index.py [--data-dir DATA_DIR]

Example:
    python scripts/build_index.py
    python scripts/build_index.py --data-dir data/sample
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
    """Build the FAISS index from evidence documents."""
    parser = argparse.ArgumentParser(
        description="Build FAISS index from cybersecurity evidence documents."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample",
        help="Directory containing evidence documents (default: data/sample)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for the FAISS index (default: data/processed/sample_faiss_index for sample data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite default research index path if building sample data",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level (default: from config)",
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

    # Guard against accidental overwrite of the full research index
    default_faiss_path = settings.resolve_path(settings.faiss_index_path)
    sample_data_path = settings.resolve_path("data/sample")
    is_sample_source = Path(args.data_dir).resolve() == sample_data_path.resolve()

    if args.output_dir:
        target_output = settings.resolve_path(args.output_dir)
    elif is_sample_source:
        target_output = settings.resolve_path("data/processed/sample_faiss_index")
        logger.info("Preserving research index; saving sample index to: %s", target_output)
    else:
        target_output = default_faiss_path

    if target_output == default_faiss_path and is_sample_source and not args.force:
        logger.warning(
            "Safeguard active: redirected sample index to 'data/processed/sample_faiss_index' to protect existing research index. Pass --force to override."
        )
        target_output = settings.resolve_path("data/processed/sample_faiss_index")

    logger.info("=" * 60)
    logger.info("CyberAI-Investigator — Build Index")
    logger.info("=" * 60)

    # Build index
    pipeline = RAGPipeline(settings)
    summary = pipeline.build_index(args.data_dir, output_path=target_output)

    # Print summary
    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
