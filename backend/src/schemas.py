"""
schemas.py
==========
Pydantic models for all API request/response objects and internal pipeline
data structures.

Using Pydantic v2 throughout.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ── API request ───────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="The question to answer")
    debug: bool = Field(False, description="Return detailed pipeline trace")


# ── Source / citation ─────────────────────────────────────────────────────────

class SourceMetadata(BaseModel):
    specification: str = Field(default="3GPP TS 23.501")
    release: str = Field(default="18")
    version: str = Field(default="V18.12.0")
    section: str
    all_sections: Optional[str] = None
    section_title: str
    parent_section: Optional[str] = None
    page_start: int
    page_end: int
    chunk_id: str

    def citation_text(self) -> str:
        pages = (f"Page {self.page_start}" if self.page_start == self.page_end
                 else f"Pages {self.page_start}–{self.page_end}")
        return (
            f"{self.specification} — "
            f"Section {self.section}: {self.section_title} ({pages})"
        )


# ── Claim verification ────────────────────────────────────────────────────────

class Claim(BaseModel):
    text: str
    source_ids: List[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    claim: str
    supported: bool
    reason: str
    source_ids: List[str] = Field(default_factory=list)


# ── Internal LLM structured output ───────────────────────────────────────────

class LLMStructuredOutput(BaseModel):
    answer: str
    claims: List[Claim] = Field(default_factory=list)
    should_abstain: bool = False
    abstain_reason: Optional[str] = None


# ── API response ──────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: Optional[str] = None
    # Optional debug info
    debug_info: Optional[Dict[str, Any]] = None


# ── Stats response ────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    indexed_chunks: int
    specification: str
    release: str
    version: str
    embedding_model: str
    reranker_model: str
    chroma_collection: str
    status: str = "ready"


# ── Health response ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    chroma_ready: bool
    bm25_ready: bool
    llm_configured: bool


# ── Retrieval debug ───────────────────────────────────────────────────────────

class RetrievalDebugInfo(BaseModel):
    original_query: str
    processed_query: str
    dense_results: List[Dict[str, Any]] = Field(default_factory=list)
    bm25_results: List[Dict[str, Any]] = Field(default_factory=list)
    rrf_scores: List[Dict[str, Any]] = Field(default_factory=list)
    reranker_scores: List[Dict[str, Any]] = Field(default_factory=list)
    selected_chunks: List[str] = Field(default_factory=list)
    evidence_gate_passed: bool = False
    evidence_gate_reason: str = ""
    generated_claims: List[str] = Field(default_factory=list)
    verification_results: List[Dict[str, Any]] = Field(default_factory=list)
    abstained: bool = False
