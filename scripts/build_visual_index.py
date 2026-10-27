#!/usr/bin/env python3
"""Build dedicated FAISS visual index from cybersecurity images using CLIP embeddings.

Processes:
- SIEM alert screenshots (PowerShell bypass)
- Terminal execution screenshots (Mimikatz logonpasswords)
- Incident attack-flow architecture diagrams
- Visual IOC threat intelligence tables

Generates:
1. OCR extraction (Observed Evidence)
2. VLM visual interpretation (Model Interpretation)
3. 512-dimensional normalized CLIP visual embeddings
4. Persistent VisualVectorStore saved to data/processed/visual_faiss_index

Usage:
    python scripts/build_visual_index.py [--output-dir data/processed/visual_faiss_index]
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

from src.multimodal.image_pipeline import VisualEvidencePipeline
from src.multimodal.visual_embedding_service import CLIPVisualEmbeddingService
from src.multimodal.visual_vector_store import VisualVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-32s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BuildVisualIndex")


def main() -> None:
    """Execute visual evidence processing and index construction."""
    parser = argparse.ArgumentParser(description="Build FAISS visual index from cybersecurity images.")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/raw/images",
        help="Directory containing raw evidence images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/visual_faiss_index",
        help="Destination directory for visual FAISS index.",
    )
    args = parser.parse_args()

    start_time = time.time()
    images_dir = PROJECT_ROOT / args.images_dir
    output_dir = PROJECT_ROOT / args.output_dir

    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    logger.info("=" * 75)
    logger.info("STARTING PHASE 3 VISUAL EVIDENCE PROCESSING & VECTOR INDEXING")
    logger.info("=" * 75)

    # 1. Pipeline execution: OCR + VLM + Metadata extraction
    pipeline = VisualEvidencePipeline()
    canonical_records = pipeline.process_directory(images_dir)

    if not canonical_records:
        logger.error("No image files found in %s", images_dir)
        sys.exit(1)

    logger.info("=" * 75)
    logger.info("PROCESSED VISUAL EVIDENCE SUMMARY:")
    for idx, rec in enumerate(canonical_records, start=1):
        meta = rec.metadata
        logger.info(
            "  [%d] %-38s | Modality: %-16s | Res: %4dx%-4d | OCR: %2d lines | ATT&CK: %s",
            idx,
            rec.source_file,
            rec.modality,
            meta.get("width", 0),
            meta.get("height", 0),
            len(meta.get("ocr_text", "").splitlines()),
            rec.attck_technique or "None",
        )
    logger.info("=" * 75)

    # 2. Generate 512-dim CLIP visual embeddings
    embedding_service = CLIPVisualEmbeddingService()
    image_paths = [rec.raw_reference for rec in canonical_records]
    logger.info("Generating 512-dimensional CLIP embeddings for %d images...", len(image_paths))
    visual_embeddings = embedding_service.embed_images(image_paths)

    # 3. Store and persist in VisualVectorStore
    visual_store = VisualVectorStore(dimension=embedding_service.dimension)
    record_dicts = [rec.to_document_dict() for rec in canonical_records]
    # Add evidence_id and raw_reference explicitly to record dicts
    for r_dict, c_rec in zip(record_dicts, canonical_records):
        r_dict["evidence_id"] = c_rec.evidence_id
        r_dict["raw_reference"] = c_rec.raw_reference
        r_dict["source_file"] = c_rec.source_file
        r_dict["modality"] = c_rec.modality

    visual_store.add_records(visual_embeddings, record_dicts)
    visual_store.save(output_dir)

    elapsed = time.time() - start_time
    logger.info("=" * 75)
    logger.info("PHASE 3 VISUAL INDEX BUILD COMPLETE in %.2f seconds", elapsed)
    logger.info("  - Total Visual Vectors Indexed: %d", visual_store.total_vectors)
    logger.info("  - Visual Embedding Dimension:   %d (CLIP ViT-B/32)", visual_store.dimension)
    logger.info("  - Visual FAISS Index Path:      %s", output_dir)
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
