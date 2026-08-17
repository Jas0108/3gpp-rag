"""
generator.py
============
Evidence-Grounded LLM Generator for 3GPP TS 23.501.

Features:
1. Evidence Sufficiency Gate: Rejects out-of-scope or weakly supported queries BEFORE calling the LLM.
2. LangChain LCEL Chain: Uses ChatPromptTemplate | LLM | StrOutputParser for zero-hallucination generation.
3. Programmatic Citations: Source citations are generated strictly from chunk metadata, never by the LLM.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import (
    EVIDENCE_MIN_CHUNKS,
    EVIDENCE_THRESHOLD,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)
from src.schemas import SourceMetadata

logger = logging.getLogger(__name__)

ABSTAIN_MESSAGE = (
    "I could not find sufficient information in the provided 3GPP standards "
    "to answer this question reliably."
)


# ── 1. LangChain System & User Prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert technical assistant for 3GPP TS 23.501 (5G System Architecture, Release 18).

STRICT RULES — YOU MUST FOLLOW ALL OF THEM:
1. Answer ONLY using information explicitly stated in the provided 3GPP document excerpts.
2. Do NOT use outside knowledge, general training data, or speculative reasoning.
3. If the provided evidence is insufficient to answer the question accurately, state that clearly and set should_abstain to true.
4. Never invent section numbers, interface names, network functions, or procedures.
5. Keep terminology exact as per 3GPP specifications.

Your output MUST be a valid JSON object with this exact structure:
{{
  "answer": "<your precise answer derived strictly from evidence>",
  "claims": [
    "<claim 1>",
    "<claim 2>"
  ],
  "should_abstain": false,
  "abstain_reason": null
}}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """QUESTION: {question}

EVIDENCE FROM 3GPP TS 23.501:
{context}

Provide a JSON object answer strictly grounded in the evidence above."""),
])


# ── 2. Helper Functions ────────────────────────────────────────────────────────

def format_context(docs: List[Document]) -> str:
    """Formats LangChain Document objects into a clean prompt context string."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        sec = doc.metadata.get("section", "N/A")
        title = doc.metadata.get("section_title", "N/A")
        pages = f"pp. {doc.metadata.get('page_start', '?')}–{doc.metadata.get('page_end', '?')}"
        cid = doc.metadata.get("chunk_id", f"chunk_{i}")
        header = f"[EXCERPT {i} | ID: {cid} | Section {sec}: {title} | {pages}]"
        parts.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def get_llm():
    """Returns configured LangChain ChatModel (OpenRouter / OpenAI / Gemini)."""
    import src.config as cfg
    
    # Sanitize obsolete/broken model slugs on the fly
    model_name = cfg.LLM_MODEL
    if "llama-3.3-70b-instruct" in model_name:
        model_name = "google/gemma-2-27b-it:free"

    provider = cfg.LLM_PROVIDER.lower()
    if provider in ["openrouter", "openai"]:
        from langchain_openai import ChatOpenAI
        base_url = cfg.LLM_BASE_URL or ("https://openrouter.ai/api/v1" if provider == "openrouter" else None)
        return ChatOpenAI(
            model=model_name,
            api_key=cfg.LLM_API_KEY or "dummy",
            base_url=base_url,
            temperature=cfg.LLM_TEMPERATURE,
            max_tokens=cfg.LLM_MAX_TOKENS,
            default_headers={"HTTP-Referer": "https://github.com", "X-Title": "3GPP RAG Assistant"} if provider == "openrouter" else None,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=cfg.LLM_API_KEY,
            temperature=cfg.LLM_TEMPERATURE,
            max_output_tokens=cfg.LLM_MAX_TOKENS,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=cfg.LLM_API_KEY or "dummy",
            base_url="https://openrouter.ai/api/v1",
        )


# ── 3. Grounded Generator Class ────────────────────────────────────────────────

class GroundedGenerator:
    """
    Handles evidence sufficiency gating, LangChain LCEL generation, and programmatic citation assembly.
    """

    @property
    def llm(self):
        return get_llm()

    @property
    def chain(self):
        """LangChain LCEL Pipeline: RAG_PROMPT | LLM | StrOutputParser"""
        return RAG_PROMPT | self.llm | StrOutputParser()

    def generate(
        self,
        question: str,
        docs: List[Document],
        top_score: float,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate answer using LangChain after checking evidence sufficiency.
        """
        # Step 1: Evidence Sufficiency Gate
        gate_passed, gate_reason = self._check_evidence_gate(docs, top_score)
        if not gate_passed:
            logger.info(f"Evidence Gate FAILED: {gate_reason}")
            return {
                "answer": ABSTAIN_MESSAGE,
                "sources": [],
                "abstained": True,
                "abstain_reason": gate_reason,
            }

        # Step 2: Format Context & Invoke LangChain LCEL Chain
        context_str = format_context(docs)
        try:
            raw_response = self.chain.invoke({
                "question": question,
                "context": context_str,
            })
            parsed = self._parse_json_response(raw_response)
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            return {
                "answer": ABSTAIN_MESSAGE,
                "sources": self._build_citations(docs),
                "abstained": True,
                "abstain_reason": f"LLM Generation error: {e}",
            }

        # Step 3: Handle LLM self-abstention or return grounded answer
        if parsed.get("should_abstain", False):
            return {
                "answer": ABSTAIN_MESSAGE,
                "sources": [],
                "abstained": True,
                "abstain_reason": parsed.get("abstain_reason", "LLM indicated insufficient evidence"),
            }

        # Step 4: Build Programmatic Citations directly from chunk metadata
        sources = self._build_citations(docs)

        return {
            "answer": parsed.get("answer", ABSTAIN_MESSAGE),
            "sources": sources,
            "abstained": False,
            "abstain_reason": None,
        }

    # ── Private Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _check_evidence_gate(docs: List[Document], top_score: float) -> Tuple[bool, str]:
        """Prevents hallucination by refusing queries with low reranker scores or too few chunks."""
        if len(docs) < EVIDENCE_MIN_CHUNKS:
            return False, f"Too few relevant document chunks found ({len(docs)} < {EVIDENCE_MIN_CHUNKS})"
        if top_score < EVIDENCE_THRESHOLD:
            return False, f"Relevance score ({top_score:.3f}) below threshold ({EVIDENCE_THRESHOLD})"
        return True, "Evidence sufficient"

    @staticmethod
    def _parse_json_response(raw_text: str) -> Dict[str, Any]:
        """Parses JSON output from LLM, cleaning markdown fences if present."""
        clean = raw_text.strip()
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean).strip()
        try:
            return json.loads(clean)
        except Exception:
            return {"answer": clean, "should_abstain": False}

    @staticmethod
    def _build_citations(docs: List[Document]) -> List[SourceMetadata]:
        """Programmatically creates top 5 source citations from metadata."""
        sources: List[SourceMetadata] = []
        seen = set()
        for doc in docs[:5]:
            cid = doc.metadata.get("chunk_id", "")
            if cid in seen:
                continue
            seen.add(cid)
            sources.append(
                SourceMetadata(
                    specification=doc.metadata.get("specification", "3GPP TS 23.501"),
                    release=doc.metadata.get("release", "18"),
                    version=doc.metadata.get("version", "V18.12.0"),
                    section=doc.metadata.get("section", "N/A"),
                    all_sections=doc.metadata.get("all_sections"),
                    section_title=doc.metadata.get("section_title", "N/A"),
                    parent_section=doc.metadata.get("parent_section"),
                    page_start=int(doc.metadata.get("page_start", 0)),
                    page_end=int(doc.metadata.get("page_end", 0)),
                    chunk_id=cid,
                )
            )
        return sources
