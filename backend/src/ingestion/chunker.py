"""
chunker.py
==========
Section-aware chunking of 3GPP TS 23.501.

Strategy
--------
1. For each Section, estimate token count (≈ chars / 4).
2. If section text fits within CHUNK_TARGET_TOKENS → one chunk.
3. If it is too long → split on paragraph / sentence boundaries
   while respecting the overlap window.
4. Every chunk gets rich metadata copied from its Section.
5. Returns LangChain Document objects so the rest of the pipeline
   can use them natively.

LangChain integration
---------------------
We return ``langchain_core.documents.Document`` objects.
This means the rest of the pipeline (embeddings, vector store, BM25)
can use LangChain-native interfaces.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import List, Optional

from langchain_core.documents import Document

from src.config import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    RELEASE,
    SPECIFICATION,
    VERSION,
)
from src.ingestion.section_parser import Section

logger = logging.getLogger(__name__)

# Approximate chars per token (conservative for technical text)
_CHARS_PER_TOKEN: int = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _make_chunk_id(section_number: str, chunk_index: int) -> str:
    """
    Create a deterministic, human-readable chunk ID.
    Example: ts23501_4_2_2_001
    """
    safe_num = re.sub(r"[^a-zA-Z0-9]", "_", section_number)
    return f"ts23501_{safe_num}_{chunk_index:03d}"


# ── Main chunker ──────────────────────────────────────────────────────────────

class SectionAwareChunker:
    """
    Converts a list of Section objects into LangChain Documents.

    Parameters
    ----------
    target_tokens   : target chunk size in tokens (~800–1200)
    overlap_tokens  : overlap between consecutive chunks
    min_chars       : discard chunks smaller than this (noise/TOC fragments)
    max_chars       : hard upper bound (safety valve)
    """

    def __init__(
        self,
        target_tokens: int = CHUNK_TARGET_TOKENS,
        overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
        min_chars: int = CHUNK_MIN_CHARS,
        max_chars: int = CHUNK_MAX_CHARS,
    ):
        self.target_chars = target_tokens * _CHARS_PER_TOKEN
        self.overlap_chars = overlap_tokens * _CHARS_PER_TOKEN
        self.min_chars = min_chars
        self.max_chars = max_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_sections(self, sections: List[Section]) -> List[Document]:
        """
        Process all sections and return a flat list of LangChain Documents.
        """
        all_docs: List[Document] = []
        chunk_counter: int = 0

        for section in sections:
            docs = self._chunk_section(section, chunk_counter)
            all_docs.extend(docs)
            chunk_counter += len(docs)

        logger.info(f"Total chunks created: {len(all_docs)}")
        return all_docs

    # ── Private helpers ───────────────────────────────────────────────────────

    def _chunk_section(
        self, section: Section, global_offset: int
    ) -> List[Document]:
        """
        Chunk a single section into one or more Documents.
        """
        text = section.text.strip()
        if not text or len(text) < self.min_chars:
            return []

        # Prepend section heading so each chunk is self-contained
        heading_prefix = f"{section.number} {section.title}\n\n"

        if _estimate_tokens(text) <= CHUNK_TARGET_TOKENS:
            # Small section → single chunk
            chunk_text = heading_prefix + text
            if len(chunk_text) > self.max_chars:
                chunk_text = chunk_text[: self.max_chars]
            doc = self._make_document(
                text=chunk_text,
                section=section,
                chunk_index=0,
                total_chunks=1,
            )
            return [doc] if doc else []

        # Large section → split into multiple chunks
        return self._split_section(section, heading_prefix)

    def _split_section(
        self, section: Section, heading_prefix: str
    ) -> List[Document]:
        """
        Split a long section body into overlapping chunks.

        Splitting strategy (in order of preference):
        1. Split on double-newline (paragraph boundary).
        2. Split on single-newline.
        3. Hard-split at max_chars (last resort).
        """
        text = section.text.strip()
        paragraphs = self._split_into_paragraphs(text)

        chunks: List[str] = []
        current_paras: List[str] = []
        current_len: int = 0

        for para in paragraphs:
            para_len = len(para)

            if current_len + para_len > self.target_chars and current_paras:
                # Flush current chunk
                chunk_body = "\n\n".join(current_paras)
                chunks.append(heading_prefix + chunk_body)

                # Overlap: keep last few paragraphs
                overlap_paras = self._get_overlap_paras(current_paras)
                current_paras = overlap_paras + [para]
                current_len = sum(len(p) for p in current_paras)
            else:
                current_paras.append(para)
                current_len += para_len

        # Flush remaining
        if current_paras:
            chunk_body = "\n\n".join(current_paras)
            chunks.append(heading_prefix + chunk_body)

        # Convert to Documents
        total = len(chunks)
        docs: List[Document] = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text) < self.min_chars:
                continue
            # Truncate at hard max
            if len(chunk_text) > self.max_chars:
                chunk_text = chunk_text[: self.max_chars]
            doc = self._make_document(
                text=chunk_text,
                section=section,
                chunk_index=i,
                total_chunks=total,
            )
            if doc:
                docs.append(doc)
        return docs

    def _get_overlap_paras(self, paras: List[str]) -> List[str]:
        """Return the last few paragraphs that fit within overlap_chars."""
        overlap_paras: List[str] = []
        total = 0
        for para in reversed(paras):
            if total + len(para) > self.overlap_chars:
                break
            overlap_paras.insert(0, para)
            total += len(para)
        return overlap_paras

    @staticmethod
    def _split_into_paragraphs(text: str) -> List[str]:
        """
        Split text into paragraphs.
        Prefer double-newline boundaries; fall back to single-newline.
        Filter out very short fragments (< 30 chars).
        """
        # Try double-newline first
        paras = re.split(r"\n\n+", text)
        result: List[str] = []
        for p in paras:
            p = p.strip()
            if len(p) >= 30:
                result.append(p)
            elif result:
                # Attach tiny fragment to previous paragraph
                result[-1] += " " + p
        return result if result else [text]

    def _make_document(
        self,
        text: str,
        section: Section,
        chunk_index: int,
        total_chunks: int,
    ) -> Optional[Document]:
        """Create a LangChain Document with full 3GPP metadata."""
        if not text.strip() or len(text) < self.min_chars:
            return None

        chunk_id = _make_chunk_id(section.number, chunk_index)

        # Extract all section numbers present in the chunk body
        found_sections = list(set(re.findall(r"\b\d+\.\d+(?:\.\d+)*\b", text)))
        if section.number and section.number not in found_sections:
            found_sections.insert(0, section.number)
        all_sections_str = ",".join(found_sections)

        metadata = {
            # 3GPP document identifiers
            "specification": SPECIFICATION,
            "release": RELEASE,
            "version": VERSION,
            # Section hierarchy
            "section": section.number,
            "all_sections": all_sections_str,
            "section_title": section.title,
            "parent_section": section.parent_number,
            "section_level": section.level,
            # Position
            "page_start": section.page_start,
            "page_end": section.page_end,
            # Chunk info
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "total_chunks_in_section": total_chunks,
            # Sizing
            "char_count": len(text),
            "approx_tokens": _estimate_tokens(text),
        }

        return Document(page_content=text, metadata=metadata)
