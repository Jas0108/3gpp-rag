"""
config.py
=========
Central configuration for the 3GPP RAG Chatbot.
All values are loaded from environment variables (with sensible defaults).
Never hard-code API keys here.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent          # backend root

# ── Load .env file ─────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv()

DATA_DIR: Path = BASE_DIR / "data"
CHROMA_DIR: Path = BASE_DIR / "chroma_db"
EVAL_DIR: Path = BASE_DIR / "evaluation"

PDF_PATH: Path = DATA_DIR / os.getenv("PDF_FILENAME", "23501-ic0.pdf")

# ── Document metadata ─────────────────────────────────────────────────────────
SPECIFICATION: str = os.getenv("SPECIFICATION", "3GPP TS 23.501")
RELEASE: str = os.getenv("RELEASE", "18")
VERSION: str = os.getenv("VERSION", "V18.12.0")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_TARGET_TOKENS: int = int(os.getenv("CHUNK_TARGET_TOKENS", "1000"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "150"))
CHUNK_MIN_CHARS: int = int(os.getenv("CHUNK_MIN_CHARS", "100"))
CHUNK_MAX_CHARS: int = int(os.getenv("CHUNK_MAX_CHARS", "6000"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "ts23501")

# ── Retrieval ─────────────────────────────────────────────────────────────────
DENSE_TOP_K: int = int(os.getenv("DENSE_TOP_K", "20"))
BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "20"))
RRF_K: int = int(os.getenv("RRF_K", "60"))           # RRF constant
HYBRID_TOP_N: int = int(os.getenv("HYBRID_TOP_N", "20"))

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "5"))

# ── Context expansion ─────────────────────────────────────────────────────────
CONTEXT_EXPAND_ENABLED: bool = os.getenv(
    "CONTEXT_EXPAND_ENABLED", "true"
).lower() == "true"
CONTEXT_EXPAND_WINDOW: int = int(os.getenv("CONTEXT_EXPAND_WINDOW", "1"))

# ── Evidence sufficiency gate ─────────────────────────────────────────────────
EVIDENCE_THRESHOLD: float = float(os.getenv("EVIDENCE_THRESHOLD", "0.15"))
EVIDENCE_MIN_CHUNKS: int = int(os.getenv("EVIDENCE_MIN_CHUNKS", "1"))

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter")   # openrouter | openai | google
LLM_MODEL: str = os.getenv("LLM_MODEL", "google/gemma-4-26b-a4b-it:free")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Debug / observability ─────────────────────────────────────────────────────
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
