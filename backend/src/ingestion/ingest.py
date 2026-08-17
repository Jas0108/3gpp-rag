"""
ingest.py
=========
Ingestion script for 3GPP TS 23.501 PDF.

Steps:
1. PDF text extraction via PyMuPDF
2. Structure-aware section parsing (handles two-line heading pattern)
3. Section-aware chunking into LangChain Documents
4. Indexing into ChromaDB (BGE Embeddings) + BM25 keyword index

Run:
    python -m src.ingestion.ingest
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PDF_PATH
from src.ingestion.chunker import SectionAwareChunker
from src.ingestion.parser import PDFExtractor
from src.ingestion.section_parser import SectionParser
from src.retrieval.retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ingestion() -> None:
    logger.info("=" * 60)
    logger.info("3GPP TS 23.501 Ingestion Pipeline")
    logger.info("=" * 60)

    # Step 1: Extract PDF Pages
    logger.info(f"\n[1/4] Extracting PDF: {PDF_PATH}")
    if not PDF_PATH.exists():
        logger.error(f"PDF not found at {PDF_PATH}")
        sys.exit(1)
    
    extractor = PDFExtractor(PDF_PATH)
    pdf_data = extractor.extract()
    logger.info(f"Extracted {pdf_data.total_pages} pages.")

    # Step 2: Parse Structure-Aware Sections
    logger.info("\n[2/4] Parsing sections (two-line heading pattern)...")
    parser = SectionParser(pdf_data.pages)
    sections = parser.parse()
    logger.info(f"Parsed {len(sections)} sections.")

    # Step 3: Chunk Sections into LangChain Documents
    logger.info("\n[3/4] Creating section-aware LangChain Documents...")
    chunker = SectionAwareChunker()
    docs = chunker.chunk_sections(sections)
    logger.info(f"Created {len(docs)} metadata-enriched chunks.")

    # Step 4: Index into ChromaDB & BM25
    logger.info("\n[4/4] Indexing into ChromaDB & BM25...")
    retriever = HybridRetriever()
    retriever.index_documents(docs)

    logger.info("\n" + "=" * 60)
    logger.info("Ingestion complete! All chunks indexed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ingestion()
