"""
main.py
=======
FastAPI backend for the 3GPP RAG Chatbot.

Endpoints:
  POST /api/chat    — Main question-answering endpoint
  GET  /api/health  — Health check
  GET  /api/stats   — Index statistics

The RAGPipeline is initialised on startup and shared across requests.
Ingestion (PDF → ChromaDB + BM25) must be run separately BEFORE starting the API.

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    API_HOST,
    API_PORT,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    RELEASE,
    RERANKER_MODEL,
    SPECIFICATION,
    VERSION,
)
from src.pipeline import RAGPipeline
from src.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StatsResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Global pipeline instance ──────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the RAG pipeline on startup."""
    global _pipeline
    logger.info("Initialising RAG pipeline ...")
    _pipeline = RAGPipeline()
    # Warm up: touch the vector store and BM25 to catch missing index errors early
    try:
        ready = _pipeline.is_ready()
        logger.info(f"Pipeline ready: {ready}")
        if not ready.get("chroma_ready"):
            logger.warning(
                "ChromaDB is empty! Run: python -m src.ingestion.ingest"
            )
        if not ready.get("bm25_ready"):
            logger.warning(
                "BM25 index missing! Run: python -m src.ingestion.ingest"
            )
        if not ready.get("llm_configured"):
            logger.warning(
                "LLM_API_KEY not set! Set it in .env before querying."
            )
    except Exception as e:
        logger.error(f"Pipeline initialisation error: {e}")
    yield
    logger.info("Shutting down ...")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="3GPP 5G Standards Assistant",
    description=(
        "Hallucination-resistant RAG chatbot for 3GPP TS 23.501 "
        "(System architecture for the 5G System, Release 18). "
        "Answers are grounded exclusively in retrieved specification text."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow any origin for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(
            status_code=503, detail="Pipeline not initialised"
        )
    return _pipeline


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    summary="Ask a question about 3GPP TS 23.501",
    response_description=(
        "Grounded answer with citations. If evidence is insufficient, "
        "the system abstains rather than hallucinating."
    ),
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main question-answering endpoint.

    Send a technical question about the 5G System architecture.
    The system retrieves relevant sections from 3GPP TS 23.501 and
    generates a grounded answer with source citations.

    - **question**: Your technical question (1–2000 chars)
    - **debug**: Set to `true` to receive detailed pipeline trace
    """
    pipeline = _get_pipeline()
    try:
        response = pipeline.query(
            question=request.question,
            debug=request.debug,
        )
        return response
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal pipeline error: {str(e)}",
        )


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health() -> HealthResponse:
    """Check if all pipeline components are ready."""
    pipeline = _get_pipeline()
    ready = pipeline.is_ready()
    return HealthResponse(
        status="ok" if all(ready.values()) else "degraded",
        chroma_ready=ready.get("chroma_ready", False),
        bm25_ready=ready.get("bm25_ready", False),
        llm_configured=ready.get("llm_configured", False),
    )


@app.get(
    "/api/stats",
    response_model=StatsResponse,
    summary="Index statistics",
)
async def stats() -> StatsResponse:
    """Return statistics about the indexed document corpus."""
    pipeline = _get_pipeline()
    try:
        count = pipeline.chunk_count()
    except Exception:
        count = 0

    return StatsResponse(
        indexed_chunks=count,
        specification=SPECIFICATION,
        release=RELEASE,
        version=VERSION,
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        chroma_collection=CHROMA_COLLECTION,
        status="ready" if count > 0 else "not_ingested",
    )


from fastapi.staticfiles import StaticFiles

# ── Serve Frontend ─────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "3GPP 5G Standards Assistant",
            "specification": f"{SPECIFICATION} {VERSION}",
            "docs": "/docs",
            "health": "/api/health",
            "chat": "/api/chat (POST)",
        }
