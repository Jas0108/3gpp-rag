"""
pipeline.py
===========
End-to-End Grounded RAG Pipeline for 3GPP TS 23.501.

Flow:
User Question -> Hybrid Retriever (Dense + BM25 + RRF + Cross-Encoder) ->
Evidence Sufficiency Gate -> LangChain LCEL Generator -> Programmatic Citations -> ChatResponse
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.generation.generator import GroundedGenerator
from src.retrieval.retriever import HybridRetriever
from src.schemas import ChatResponse

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG orchestrator linking retrieval and grounded generation.
    """

    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = GroundedGenerator()
        self._warmup()

    def _warmup(self):
        """Warm up vector store, BM25, and Cross-Encoder weights on initial engine startup."""
        try:
            _ = self.retriever.vector_store
            _ = self.retriever.bm25
            _ = self.retriever.reranker
        except Exception as e:
            logger.warning(f"Warmup warning: {e}")

    def query(self, question: str, debug: bool = False) -> ChatResponse:
        """
        Process a user question through the grounded RAG pipeline.
        """
        logger.info(f"Processing Query: '{question}'")

        # 1. High-precision Hybrid Retrieval + Reranking + Context Expansion
        retrieval_res = self.retriever.retrieve(question, debug=debug)
        docs = retrieval_res["documents"]
        top_score = retrieval_res["top_score"]

        # 2. Evidence Sufficiency Gate + LangChain LCEL Generation + Programmatic Citations
        gen_res = self.generator.generate(
            question=question,
            docs=docs,
            top_score=top_score,
            debug=debug,
        )

        # 3. Assemble Response
        debug_info: Dict[str, Any] = {}
        if debug:
            debug_info["retrieval"] = retrieval_res.get("debug", {})
            debug_info["top_score"] = top_score
            debug_info["retrieved_chunks"] = len(docs)

        return ChatResponse(
            answer=gen_res["answer"],
            sources=gen_res["sources"],
            abstained=gen_res["abstained"],
            abstain_reason=gen_res.get("abstain_reason"),
            debug_info=debug_info if debug else None,
        )

    def is_ready(self) -> Dict[str, bool]:
        """Status check for ChromaDB, BM25 index, and LLM configuration."""
        from src.config import LLM_API_KEY
        from src.retrieval.retriever import BM25_INDEX_PATH
        chroma_ok = False
        try:
            chroma_ok = self.retriever.vector_store._collection.count() > 0
        except Exception:
            pass

        return {
            "chroma_ready": chroma_ok,
            "bm25_ready": BM25_INDEX_PATH.exists(),
            "llm_configured": bool(LLM_API_KEY and LLM_API_KEY.strip()),
        }

    def chunk_count(self) -> int:
        """Returns the total number of indexed document chunks in ChromaDB."""
        try:
            return self.retriever.vector_store._collection.count()
        except Exception:
            return 0
