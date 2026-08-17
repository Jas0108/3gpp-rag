"""
retriever.py
============
Unified, high-precision retrieval module for 3GPP TS 23.501.

Combines:
1. Dense Retrieval (ChromaDB + HuggingFace BGE Embeddings)
2. Sparse Retrieval (BM25 for exact 3GPP acronym matching like AMF, SMF, 5QI, S-NSSAI)
3. Reciprocal Rank Fusion (RRF) for score merging
4. Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2) for precision filtering
5. Section-Aware Context Expansion (includes adjacent chunks from the same section)

All retrived items are native LangChain `Document` objects.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import (
    BASE_DIR,
    BM25_TOP_K,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CONTEXT_EXPAND_ENABLED,
    CONTEXT_EXPAND_WINDOW,
    DENSE_TOP_K,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    HYBRID_TOP_N,
    RERANKER_MODEL,
    RERANKER_TOP_K,
    RRF_K,
)

logger = logging.getLogger(__name__)
BM25_INDEX_PATH = BASE_DIR / "bm25_index.pkl"


# ── 1. Embeddings Factory ──────────────────────────────────────────────────────

def get_embedding_function():
    """Returns LangChain HuggingFaceEmbeddings using BAAI/bge-base-en-v1.5 with BGE query instruction."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


# ── 2. BM25 Tokenizer & Utilities ─────────────────────────────────────────────

def bm25_tokenize(text: str) -> List[str]:
    """Tokenizes text preserving section numbers (5.15.2) and 3GPP acronyms (AMF, S-NSSAI, 5QI, N11)."""
    sections = [s.lower() for s in re.findall(r"\b\d+\.\d+(?:\.\d+)*\b", text)]
    words = re.split(r"[\s,\.;:\(\)\[\]{}'\"]+", text.lower())
    words = [t for t in words if len(t) >= 2]
    return list(set(sections + words))


# ── 3. Hybrid Retriever (Dense + BM25 + RRF + Cross-Encoder) ──────────────────

class HybridRetriever:
    """
    Unified retriever that combines ChromaDB dense search and BM25 keyword search,
    merges rankings using Reciprocal Rank Fusion (RRF), and re-scores candidates
    with a Cross-Encoder reranker.
    """

    def __init__(self):
        self._vector_store = None
        self._bm25 = None
        self._bm25_docs: List[Document] = []
        self._reranker = None

    # ── Lazy Component Loaders ─────────────────────────────────────────────────

    @property
    def vector_store(self):
        if self._vector_store is None:
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma
            Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
            self._vector_store = Chroma(
                collection_name=CHROMA_COLLECTION,
                embedding_function=get_embedding_function(),
                persist_directory=str(CHROMA_DIR),
            )
        return self._vector_store

    @property
    def bm25(self) -> Optional[BM25Okapi]:
        if self._bm25 is None and BM25_INDEX_PATH.exists():
            with open(BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._bm25_docs = data["docs"]
        return self._bm25

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading Cross-Encoder model: {RERANKER_MODEL}")
            self._reranker = CrossEncoder(RERANKER_MODEL)
        return self._reranker

    # ── Indexing Methods (Used during Ingestion) ──────────────────────────────

    def index_documents(self, docs: List[Document], batch_size: int = 100) -> None:
        """Indexes LangChain Document objects into ChromaDB and BM25."""
        logger.info(f"Indexing {len(docs)} documents into ChromaDB...")
        ids = [doc.metadata.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(docs)]
        
        # Clear existing Chroma collection
        try:
            self.vector_store.delete_collection()
            self._vector_store = None
        except Exception:
            pass

        # Add to Chroma in batches
        store = self.vector_store
        for i in range(0, len(docs), batch_size):
            store.add_documents(documents=docs[i : i + batch_size], ids=ids[i : i + batch_size])

        # Build BM25 index
        logger.info(f"Building BM25 index over {len(docs)} documents...")
        tokenized = [bm25_tokenize(d.page_content) for d in docs]
        bm25_obj = BM25Okapi(tokenized)

        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"docs": docs, "bm25": bm25_obj}, f)

        self._bm25 = bm25_obj
        self._bm25_docs = docs
        logger.info("Indexing complete (ChromaDB + BM25).")

    # ── Retrieval & Reranking ──────────────────────────────────────────────────

    def retrieve(
        self, query: str, top_k: int = RERANKER_TOP_K, debug: bool = False
    ) -> Dict[str, Any]:
        """
        Main retrieval pipeline:
        Query -> Sub-Query Generation -> Dense & BM25 Multi-Retrieval -> RRF Merge -> Cross-Encoder Rerank -> Context Expansion
        """
        # 1. Generate Sub-Queries for Multi-Aspect / Comparison Queries
        sub_queries = self._generate_sub_queries(query)

        all_candidate_lists: List[List[Document]] = []

        for sq in sub_queries:
            # 2. Dense Retrieval (ChromaDB)
            dense_results = self.vector_store.similarity_search_with_relevance_scores(
                sq, k=DENSE_TOP_K
            )
            dense_docs = [doc for doc, _ in dense_results]
            all_candidate_lists.append(dense_docs)

            # 3. Sparse Retrieval (BM25)
            if self.bm25:
                scores = self.bm25.get_scores(bm25_tokenize(sq))
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_TOP_K]
                bm25_docs = [self._bm25_docs[i] for i in ranked_indices]
                all_candidate_lists.append(bm25_docs)

        # 4. Reciprocal Rank Fusion (RRF) across all sub-query candidate lists
        fused = self._reciprocal_rank_fusion(all_candidate_lists, k=RRF_K, top_n=HYBRID_TOP_N)

        # 5. Cross-Encoder Reranking against original user query
        reranked_docs, reranker_scores = self._rerank(query, fused, top_k=top_k)

        # 6. Context Expansion (Add adjacent chunks from same section)
        final_docs = self._expand_context(reranked_docs)

        result = {
            "documents": final_docs,
            "reranker_scores": reranker_scores,
            "top_score": max(reranker_scores) if reranker_scores else -99.0,
        }

        if debug:
            result["debug"] = {
                "query": query,
                "sub_queries": sub_queries,
                "candidate_lists": len(all_candidate_lists),
                "fused_count": len(fused),
                "reranker_scores": reranker_scores,
            }

        return result

    # ── Private Helper Functions ──────────────────────────────────────────────

    @staticmethod
    def _expand_query(query: str) -> str:
        """Appends full names for common 3GPP acronyms to improve recall."""
        expansions = {
            "AMF": "Access and Mobility Management Function",
            "SMF": "Session Management Function",
            "UPF": "User Plane Function",
            "PCF": "Policy Control Function",
            "UDM": "Unified Data Management",
            "UDR": "Unified Data Repository",
            "AUSF": "Authentication Server Function",
            "NRF": "Network Repository Function",
            "NSSF": "Network Slice Selection Function",
            "5QI": "5G QoS Indicator",
            "QOS": "Quality of Service",
            "GBR": "Guaranteed Bit Rate",
            "S-NSSAI": "Single Network Slice Selection Assistance Information",
            "NSSAI": "Network Slice Selection Assistance Information",
            "SSC": "Session and Service Continuity",
            "LBO": "Local Breakout roaming",
            "HR": "Home Routed roaming",
            "PDU": "Protocol Data Unit",
        }
        added = []
        q_upper = query.upper()
        for acronym, full in expansions.items():
            if acronym in q_upper and full.lower() not in query.lower():
                added.append(full)
        return f"{query} {' '.join(added)}".strip()

    @staticmethod
    def _generate_sub_queries(query: str) -> List[str]:
        """Generates sub-queries for complex comparison or multi-aspect technical questions."""
        queries = [query]
        expanded = HybridRetriever._expand_query(query)
        if expanded != query:
            queries.append(expanded)

        # Comparison / Difference decomposition
        diff_match = re.search(r"difference\s+between\s+(.+?)\s+and\s+(.+)", query, re.IGNORECASE)
        if diff_match:
            part1, part2 = diff_match.group(1).strip(" ?."), diff_match.group(2).strip(" ?.")
            queries.append(f"role and functions of {part1}")
            queries.append(f"role and functions of {part2}")

        # Multi-concept query splitting on 'and' or 'vs'
        if " vs " in query.lower():
            parts = re.split(r"\s+vs\s+", query, flags=re.IGNORECASE)
            for p in parts:
                queries.append(p.strip())

        return list(dict.fromkeys(queries))

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[Document]], k: int = 60, top_n: int = 20
    ) -> List[Document]:
        """Merges multiple ranked document lists using RRF."""
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for r_list in ranked_lists:
            for rank, doc in enumerate(r_list, start=1):
                cid = doc.metadata.get("chunk_id", doc.page_content[:50])
                scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
                doc_map[cid] = doc

        sorted_cids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        return [doc_map[cid] for cid in sorted_cids[:top_n]]

    def _rerank(
        self, query: str, candidates: List[Document], top_k: int = 5
    ) -> Tuple[List[Document], List[float]]:
        """Reranks document candidates using Cross-Encoder."""
        if not candidates:
            return [], []

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = [float(s) for s in self.reranker.predict(pairs)]

        # Pair candidates with scores and sort descending
        scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
        top_docs = [doc for doc, _ in scored_candidates]
        top_scores = [score for _, score in scored_candidates]

        return top_docs, top_scores

    def _expand_context(self, top_docs: List[Document]) -> List[Document]:
        """Includes adjacent chunks from the same section if context expansion is enabled."""
        if not CONTEXT_EXPAND_ENABLED or not self._bm25_docs:
            return top_docs

        # Index all corpus docs by (section, chunk_index)
        section_map: Dict[str, List[Document]] = {}
        for doc in self._bm25_docs:
            sec = doc.metadata.get("section", "")
            section_map.setdefault(sec, []).append(doc)

        # Sort section lists by chunk_index
        for sec in section_map:
            section_map[sec].sort(key=lambda d: d.metadata.get("chunk_index", 0))

        result: List[Document] = []
        seen_cids = set()

        for doc in top_docs:
            sec = doc.metadata.get("section", "")
            cidx = doc.metadata.get("chunk_index", 0)

            # Retrieve section neighbors
            sec_docs = section_map.get(sec, [])
            for neighbor in sec_docs:
                n_cidx = neighbor.metadata.get("chunk_index", 0)
                if abs(n_cidx - cidx) <= CONTEXT_EXPAND_WINDOW:
                    cid = neighbor.metadata.get("chunk_id")
                    if cid not in seen_cids:
                        seen_cids.add(cid)
                        result.append(neighbor)

        return result
