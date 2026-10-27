#!/usr/bin/env python3
"""Dataset Inventory Script for CyberAI-Investigator.

Scans the data repository (data/raw, data/processed, data/sample),
inspects each file, determines its modality, and reports:
- Dataset and file name
- File extension and modality
- File size
- Record count, columns, or event counts where applicable
- Structural and multimodal metadata (pages, tables, image dimensions)

Usage:
    python scripts/inventory_datasets.py
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

# Configure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.WARNING)


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def inspect_evtx(file_path: Path) -> dict[str, Any]:
    """Inspect an EVTX file to count events and extract unique Event IDs."""
    try:
        import Evtx.Evtx as evtx
        import xml.etree.ElementTree as ET
        import re

        event_ids = set()
        count = 0
        with evtx.Evtx(str(file_path)) as log:
            for record in log.records():
                count += 1
                try:
                    xml_str = record.xml()
                    clean_xml = re.sub(r'\sxmlns="[^"]+"', "", xml_str, count=1)
                    root = ET.fromstring(clean_xml)
                    system = root.find("System")
                    if system is not None:
                        eid = system.find("EventID")
                        if eid is not None and eid.text:
                            event_ids.add(eid.text)
                except Exception:
                    pass
        return {
            "record_count": count,
            "event_ids": sorted(list(event_ids)),
        }
    except Exception as e:
        return {"error": str(e)}


def inspect_json(file_path: Path) -> dict[str, Any]:
    """Inspect a JSON or NDJSON file."""
    # Check if NDJSON (first 5 lines valid JSON objects)
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
            if first_line.startswith("{") and not first_line.endswith(","):
                # Potential NDJSON
                count = 1
                keys = set(json.loads(first_line).keys())
                for line in f:
                    if line.strip():
                        count += 1
                return {
                    "format": "NDJSON",
                    "record_count": count,
                    "top_keys": sorted(list(keys))[:8],
                }
    except Exception:
        pass

    # Standard JSON
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
            if isinstance(data, list):
                keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
                return {
                    "format": "JSON_Array",
                    "record_count": len(data),
                    "columns": keys[:8],
                }
            elif isinstance(data, dict):
                # Check for special collections (STIX or NVD)
                if "objects" in data:
                    return {
                        "format": "STIX_2.1_Bundle",
                        "object_count": len(data["objects"]),
                        "spec_version": data.get("spec_version", "2.1"),
                    }
                elif "vulnerabilities" in data:
                    return {
                        "format": "NVD_CVE_Feed",
                        "record_count": len(data["vulnerabilities"]),
                    }
                return {
                    "format": "JSON_Object",
                    "keys": list(data.keys())[:10],
                }
    except Exception as e:
        return {"error": str(e)}

    return {}


def inspect_csv(file_path: Path) -> dict[str, Any]:
    """Inspect a CSV file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            count = sum(1 for _ in reader)
            return {
                "record_count": count,
                "column_count": len(header),
                "columns_sample": [h.strip() for h in header[:5]],
            }
    except Exception as e:
        return {"error": str(e)}


def inspect_pdf(file_path: Path) -> dict[str, Any]:
    """Inspect a PDF file."""
    try:
        import pymupdf
        doc = pymupdf.open(str(file_path))
        pages = len(doc)
        doc.close()
        return {"page_count": pages}
    except Exception as e:
        return {"error": str(e)}


def inspect_docx(file_path: Path) -> dict[str, Any]:
    """Inspect a DOCX file."""
    try:
        import docx
        doc = docx.Document(str(file_path))
        return {
            "paragraph_count": len(doc.paragraphs),
            "table_count": len(doc.tables),
        }
    except Exception as e:
        return {"error": str(e)}


def inspect_image(file_path: Path) -> dict[str, Any]:
    """Inspect an image file."""
    try:
        from PIL import Image
        with Image.open(str(file_path)) as img:
            return {
                "dimensions": f"{img.width}x{img.height}",
                "image_format": img.format,
                "mode": img.mode,
            }
    except Exception as e:
        return {"error": str(e)}


def determine_modality(path: Path) -> str:
    """Determine evidence modality from path and extension."""
    ext = path.suffix.lower()
    p_str = str(path).lower().replace("\\", "/")

    if ext == ".evtx":
        return "EVTX (Host Log)"
    elif "cic_ids" in p_str or "firewall" in p_str:
        return "Network Flow / Log"
    elif ext == ".csv":
        return "CSV / Tabular"
    elif "mitre_attack" in p_str:
        return "MITRE ATT&CK STIX"
    elif "nvd" in p_str:
        return "NVD Vulnerability"
    elif "mordor" in p_str or ext == ".json":
        return "JSON / Host Events"
    elif ext == ".pdf":
        return "PDF Document"
    elif ext == ".docx":
        return "DOCX Document"
    elif ext in {".png", ".jpg", ".jpeg"}:
        if "screenshot" in p_str:
            return "Image (Screenshot)"
        elif "diagram" in p_str:
            return "Image (Diagram)"
        elif "table" in p_str:
            return "Image (Table)"
        return "Image (Visual)"
    elif ext in {".txt", ".log"}:
        return "Text / Log"
    return "Binary / Other"


def main() -> None:
    """Scan and print full dataset inventory."""
    data_dir = PROJECT_ROOT / "data"
    print("=" * 105)
    print("CYBERAI-INVESTIGATOR: MULTIMODAL RESEARCH DATA INVENTORY")
    print("=" * 105)
    print(f"{'Dataset / Folder':<22} | {'File Name':<42} | {'Modality':<18} | {'Size':<10} | Details")
    print("-" * 105)

    categories = [
        "raw/atomic_evtx",
        "raw/mordor",
        "raw/cic_ids2017",
        "raw/mitre_attack",
        "raw/nvd",
        "raw/documents/pdf",
        "raw/documents/docx",
        "raw/images/screenshots",
        "raw/images/diagrams",
        "raw/images/tables",
        "sample",
    ]

    total_files = 0
    total_bytes = 0

    for cat in categories:
        cat_dir = data_dir / cat
        if not cat_dir.exists():
            continue

        files = sorted(
            [f for f in cat_dir.iterdir() if f.is_file() and not f.name.endswith(".gitkeep") and f.name != "SOURCE.md"]
        )

        for f in files:
            total_files += 1
            f_size = f.stat().st_size
            total_bytes += f_size
            size_str = format_size(f_size)
            modality = determine_modality(f)

            details_str = ""
            ext = f.suffix.lower()

            if ext == ".evtx":
                info = inspect_evtx(f)
                if "record_count" in info:
                    eids = ",".join(info.get("event_ids", []))
                    details_str = f"Events: {info['record_count']} (EventIDs: {eids})"
                else:
                    details_str = f"Error: {info.get('error', '')}"
            elif ext == ".csv":
                info = inspect_csv(f)
                if "record_count" in info:
                    details_str = f"Rows: {info['record_count']}, Cols: {info['column_count']}"
            elif ext == ".json":
                info = inspect_json(f)
                if "record_count" in info:
                    details_str = f"Records: {info['record_count']} ({info.get('format', '')})"
                elif "object_count" in info:
                    details_str = f"STIX Objects: {info['object_count']}"
            elif ext == ".pdf":
                info = inspect_pdf(f)
                details_str = f"Pages: {info.get('page_count', '?')}"
            elif ext == ".docx":
                info = inspect_docx(f)
                details_str = f"Paragraphs: {info.get('paragraph_count', '?')}, Tables: {info.get('table_count', '?')}"
            elif ext in {".png", ".jpg", ".jpeg"}:
                info = inspect_image(f)
                details_str = f"{info.get('dimensions', '?')}, Format: {info.get('image_format', '?')}"
            elif ext in {".zip"}:
                details_str = "Archive (original raw artifact)"

            folder_display = cat.replace("raw/", "")
            file_display = f.name if len(f.name) <= 40 else f.name[:37] + "..."
            print(f"{folder_display:<22} | {file_display:<42} | {modality:<18} | {size_str:<10} | {details_str}")

    print("=" * 105)
    print(f"Total Datasets Scanned: {len(categories)} categories")
    print(f"Total Research Files:   {total_files} files")
    print(f"Total Raw Volume:       {format_size(total_bytes)}")
    print("=" * 105)


if __name__ == "__main__":
    main()
