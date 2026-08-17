"""
parser.py
=========
Low-level PDF extraction layer.

Responsibilities:
- Open the PDF with PyMuPDF (fitz) for reliable text + page-number extraction.
- Return a list of RawPage objects: { page_number, text, blocks }.
- Also use pdfplumber as a fallback / table extraction aid.

This module does NOT know anything about sections or chunks.
Section detection happens in section_parser.py.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RawPage:
    """One page worth of extracted content."""
    page_number: int          # 1-indexed (matches PDF page label)
    text: str                 # Full plain-text of the page
    blocks: List[dict] = field(default_factory=list)  # PyMuPDF text blocks


@dataclass
class ExtractionResult:
    """Result of extracting an entire PDF."""
    pages: List[RawPage]
    total_pages: int
    pdf_path: str
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


# ── Main extractor ────────────────────────────────────────────────────────────

class PDFExtractor:
    """
    Extracts text from a 3GPP PDF using PyMuPDF.

    Strategy:
    1. Iterate pages, extract text with PyMuPDF (preserves layout reasonably).
    2. Clean up common PDF artefacts (ligatures, hyphenation, headers/footers).
    3. Optionally extract table text via pdfplumber for pages flagged as tables.
    """

    # 3GPP running header/footer patterns to strip
    _HEADER_FOOTER_PATTERNS: List[str] = [
        r"3GPP TS \d+\.\d+ V[\d\.]+ \(\d{4}-\d{2}\)",
        r"Release \d+",
        r"ETSI",
        r"^\s*\d+\s*$",            # lone page number line
        r"^3GPP$",
    ]

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self) -> ExtractionResult:
        """Extract all pages from the PDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("Install PyMuPDF: pip install pymupdf")

        logger.info(f"Opening PDF: {self.pdf_path}")
        doc = fitz.open(str(self.pdf_path))
        total_pages = doc.page_count
        logger.info(f"Total pages in PDF: {total_pages}")

        pages: List[RawPage] = []
        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_number = page_idx + 1  # 1-indexed

            # Extract text preserving layout
            raw_text = page.get_text("text")

            # Extract blocks for structural analysis
            blocks = self._extract_blocks(page)

            # Clean up text
            cleaned = self._clean_text(raw_text, page_number)

            pages.append(RawPage(
                page_number=page_number,
                text=cleaned,
                blocks=blocks,
            ))

            if page_number % 50 == 0:
                logger.info(f"  Extracted page {page_number}/{total_pages}")

        doc.close()
        logger.info(f"Extraction complete. {len(pages)} pages extracted.")

        return ExtractionResult(
            pages=pages,
            total_pages=total_pages,
            pdf_path=str(self.pdf_path),
            metadata=self._extract_pdf_metadata(),
        )

    def extract_page_range(
        self, start: int, end: int
    ) -> List[RawPage]:
        """Extract a specific page range (1-indexed, inclusive)."""
        result = self.extract()
        return [p for p in result.pages if start <= p.page_number <= end]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_blocks(self, page) -> List[dict]:
        """Extract text blocks with bounding box info for layout analysis."""
        raw_blocks = page.get_text("blocks")
        clean = []
        for b in raw_blocks:
            # b = (x0, y0, x1, y1, text, block_no, block_type)
            if len(b) >= 5 and b[4].strip():
                clean.append({
                    "x0": round(b[0], 1),
                    "y0": round(b[1], 1),
                    "x1": round(b[2], 1),
                    "y1": round(b[3], 1),
                    "text": b[4].strip(),
                    "block_no": b[5] if len(b) > 5 else -1,
                })
        return clean

    def _clean_text(self, text: str, page_number: int) -> str:
        """
        Remove PDF artefacts common in 3GPP documents:
        - running headers / footers
        - ligature characters (ﬁ → fi, etc.)
        - excessive blank lines
        """
        # Fix common ligatures
        ligature_map = {
            "\ufb01": "fi",
            "\ufb02": "fl",
            "\ufb00": "ff",
            "\ufb03": "ffi",
            "\ufb04": "ffl",
            "\u2013": "-",
            "\u2014": "--",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2022": "-",   # bullet → dash
            "\u00a0": " ",   # non-breaking space
        }
        for bad, good in ligature_map.items():
            text = text.replace(bad, good)

        lines = text.split("\n")
        cleaned_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            # Remove running headers/footers
            if self._is_header_footer(stripped):
                continue
            cleaned_lines.append(stripped)

        # Collapse 3+ blank lines into 2
        result = "\n".join(cleaned_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def _is_header_footer(self, line: str) -> bool:
        """Return True if a line looks like a 3GPP running header/footer."""
        if not line:
            return False
        for pattern in self._HEADER_FOOTER_PATTERNS:
            if re.fullmatch(pattern, line):
                return True
        # 3GPP typical header: "3GPP TS 23.501 version 18..."
        if re.match(r"^3GPP TS \d+\.\d+", line):
            return True
        if re.match(r"^ETSI\s+TS\s+\d+", line):
            return True
        return False

    def _extract_pdf_metadata(self) -> dict:
        """Extract PDF document metadata."""
        try:
            import fitz
            doc = fitz.open(str(self.pdf_path))
            meta = doc.metadata
            doc.close()
            return {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creator": meta.get("creator", ""),
                "creation_date": meta.get("creationDate", ""),
            }
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
            return {}
