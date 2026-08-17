"""
metadata.py
===========
Metadata utilities for 3GPP document chunks.

Provides:
- Canonical metadata schema (as a TypedDict / dataclass)
- Metadata normalisation helpers
- Chunk ID generation
"""

from __future__ import annotations

from typing import TypedDict, Optional


class ChunkMetadata(TypedDict):
    """Canonical metadata carried by every LangChain Document chunk."""
    # 3GPP identifiers
    specification: str          # e.g. "3GPP TS 23.501"
    release: str                # e.g. "18"
    version: str                # e.g. "V18.12.0"
    # Section hierarchy
    section: str                # e.g. "4.2.2"
    section_title: str          # e.g. "Network Functions and entities"
    parent_section: str         # e.g. "4.2"
    section_level: int          # 1, 2, 3, …
    # Page range
    page_start: int
    page_end: int
    # Chunk identifiers
    chunk_id: str               # e.g. "ts23501_4_2_2_001"
    chunk_index: int            # 0-indexed within section
    total_chunks_in_section: int
    # Sizing
    char_count: int
    approx_tokens: int


def validate_metadata(meta: dict) -> list[str]:
    """
    Return a list of missing required fields.
    An empty list means the metadata is complete.
    """
    required = [
        "specification", "release", "version",
        "section", "section_title", "page_start", "page_end", "chunk_id",
    ]
    return [f for f in required if not meta.get(f)]


def format_citation(meta: dict) -> str:
    """
    Generate a human-readable citation string from chunk metadata.
    Used by the citation system to create programmatic (not LLM-generated) cites.
    """
    spec  = meta.get("specification", "3GPP TS 23.501")
    sec   = meta.get("section", "")
    title = meta.get("section_title", "")
    p_s   = meta.get("page_start", "?")
    p_e   = meta.get("page_end", "?")
    page_str = f"Page {p_s}" if p_s == p_e else f"Pages {p_s}–{p_e}"
    return f"{spec} — Section {sec}: {title} ({page_str})"
