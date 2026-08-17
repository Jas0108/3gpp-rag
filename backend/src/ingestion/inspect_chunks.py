"""
inspect_chunks.py
=================
Phase 4 — Chunk Validation / Debug Script

Run with:
    python -m src.ingestion.inspect_chunks

Prints:
- Total pages extracted
- Total sections detected
- Section tree (depth-limited)
- Total chunk count
- Chunk size statistics (min/max/mean/median tokens)
- Section distribution (how many chunks per top-level section)
- Sample chunks (first + a random sample)
- Metadata snapshot
- Warnings: missing metadata, too-small or too-large chunks

Goal: manually verify the parser before generating embeddings.
"""

from __future__ import annotations

import io
import logging
import statistics
import sys
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Ensure project root is importable ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import PDF_PATH, CHUNK_MIN_CHARS, CHUNK_MAX_CHARS
from src.ingestion.parser import PDFExtractor
from src.ingestion.section_parser import SectionParser
from src.ingestion.chunker import SectionAwareChunker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── ANSI colours for terminal ─────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"


def header(title: str) -> None:
    width = 70
    print(f"\n{C.BOLD}{C.CYAN}{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}{C.RESET}")


def warn(msg: str) -> None:
    print(f"{C.YELLOW}  [!]  {msg}{C.RESET}")


def ok(msg: str) -> None:
    print(f"{C.GREEN}  [OK] {msg}{C.RESET}")


def error(msg: str) -> None:
    print(f"{C.RED}  [ERR] {msg}{C.RESET}")


# ── Main inspection routine ───────────────────────────────────────────────────

def run_inspection() -> None:
    # ── 1. PDF extraction ─────────────────────────────────────────────────────
    header("PHASE 1 — PDF Extraction")

    if not PDF_PATH.exists():
        error(f"PDF not found: {PDF_PATH}")
        sys.exit(1)

    extractor = PDFExtractor(PDF_PATH)
    extraction = extractor.extract()

    print(f"\n  PDF path     : {extraction.pdf_path}")
    print(f"  Total pages  : {extraction.total_pages}")
    print(f"  PDF metadata : {extraction.metadata}")

    # Quick sanity: total characters extracted
    total_chars = sum(len(p.text) for p in extraction.pages)
    print(f"  Total chars  : {total_chars:,}")
    if total_chars < 50_000:
        warn("Very few characters extracted — PDF may be scanned/image-based.")
    else:
        ok(f"Extracted {total_chars:,} characters.")

    # Show first non-empty page content
    print(f"\n{C.BOLD}  Sample — Page 1 text (first 500 chars):{C.RESET}")
    for page in extraction.pages[:5]:
        if len(page.text) > 100:
            print(f"\n  --- Page {page.page_number} ---")
            print("  " + page.text[:500].replace("\n", "\n  "))
            break

    # ── 2. Section parsing ────────────────────────────────────────────────────
    header("PHASE 2 — Section Parsing")

    parser = SectionParser(extraction.pages)
    sections = parser.parse()

    print(f"\n  Total sections detected : {len(sections)}")

    if not sections:
        error("No sections detected! Check section_parser.py regex patterns.")
        sys.exit(1)

    # Level distribution
    by_level: dict = {}
    for s in sections:
        by_level.setdefault(s.level, []).append(s)

    print(f"\n  Sections by depth level:")
    for level in sorted(by_level.keys()):
        print(f"    Level {level}: {len(by_level[level])} sections")

    # Top-level sections
    top_level = [s for s in sections if s.level == 1]
    print(f"\n  Top-level sections ({len(top_level)}):")
    for s in top_level[:25]:
        print(f"    [{s.number}] {s.title}  "
              f"(pp. {s.page_start}–{s.page_end}, {s.word_count} words)")

    # Section tree (depth ≤ 2)
    print(f"\n{C.BOLD}  Section tree (levels 1–2):{C.RESET}")
    parser.print_tree(max_depth=2)

    # ── 3. Chunking ───────────────────────────────────────────────────────────
    header("PHASE 3 — Chunking")

    chunker = SectionAwareChunker()
    docs = chunker.chunk_sections(sections)

    print(f"\n  Total chunks: {len(docs)}")

    if not docs:
        error("No chunks produced! Check chunker.py.")
        sys.exit(1)

    # ── 4. Size statistics ────────────────────────────────────────────────────
    header("PHASE 4 — Chunk Size Statistics")

    token_counts = [d.metadata.get("approx_tokens", 0) for d in docs]
    char_counts  = [d.metadata.get("char_count", 0) for d in docs]

    print(f"\n  Token statistics (approx):")
    print(f"    Min     : {min(token_counts)}")
    print(f"    Max     : {max(token_counts)}")
    print(f"    Mean    : {statistics.mean(token_counts):.0f}")
    print(f"    Median  : {statistics.median(token_counts):.0f}")
    print(f"    Stdev   : {statistics.stdev(token_counts):.0f}")

    print(f"\n  Char statistics:")
    print(f"    Min     : {min(char_counts)}")
    print(f"    Max     : {max(char_counts)}")
    print(f"    Mean    : {statistics.mean(char_counts):.0f}")

    # Distribution buckets
    buckets = {"<200": 0, "200-800": 0, "800-1600": 0,
               "1600-3200": 0, ">3200": 0}
    for t in token_counts:
        if t < 200:       buckets["<200"] += 1
        elif t < 800:     buckets["200-800"] += 1
        elif t < 1600:    buckets["800-1600"] += 1
        elif t < 3200:    buckets["1600-3200"] += 1
        else:             buckets[">3200"] += 1

    print(f"\n  Token distribution:")
    for bucket, count in buckets.items():
        bar = "█" * (count // max(1, len(docs) // 40))
        print(f"    {bucket:>12} tokens : {count:>4}  {bar}")

    # ── 5. Section distribution ───────────────────────────────────────────────
    header("PHASE 5 — Section Distribution")

    sec_counts: dict = {}
    for doc in docs:
        sec = doc.metadata.get("section", "UNKNOWN")
        # Group by top-level section
        top = sec.split(".")[0] if sec else "UNKNOWN"
        sec_counts[top] = sec_counts.get(top, 0) + 1

    print(f"\n  Chunks per top-level section:")
    for sec_num in sorted(sec_counts.keys(), key=lambda x: (
        int(x) if x.isdigit() else 999, x
    )):
        bar = "█" * sec_counts[sec_num]
        print(f"    Section {sec_num:>4}: {sec_counts[sec_num]:>4} chunks  {bar[:60]}")

    # ── 6. Metadata validation ────────────────────────────────────────────────
    header("PHASE 6 — Metadata Validation")

    required_fields = [
        "specification", "release", "version", "section",
        "section_title", "page_start", "page_end", "chunk_id",
    ]

    missing_meta: list = []
    tiny_chunks:  list = []
    huge_chunks:  list = []

    for i, doc in enumerate(docs):
        m = doc.metadata
        missing = [f for f in required_fields if not m.get(f)]
        if missing:
            missing_meta.append((i, m.get("chunk_id", f"idx_{i}"), missing))
        if m.get("approx_tokens", 0) < 30:
            tiny_chunks.append((i, m.get("chunk_id"), m.get("approx_tokens")))
        if m.get("approx_tokens", 0) > 2000:
            huge_chunks.append((i, m.get("chunk_id"), m.get("approx_tokens")))

    if missing_meta:
        warn(f"{len(missing_meta)} chunks have missing metadata fields:")
        for idx, cid, fields in missing_meta[:10]:
            print(f"    chunk [{cid}] missing: {fields}")
    else:
        ok("All chunks have complete metadata.")

    if tiny_chunks:
        warn(f"{len(tiny_chunks)} chunks are very small (<30 tokens):")
        for idx, cid, tok in tiny_chunks[:10]:
            print(f"    [{cid}] — {tok} tokens")
    else:
        ok("No excessively small chunks.")

    if huge_chunks:
        warn(f"{len(huge_chunks)} chunks exceed 2000 tokens:")
        for idx, cid, tok in huge_chunks[:10]:
            print(f"    [{cid}] — {tok} tokens")
    else:
        ok("No excessively large chunks.")

    # ── 7. Sample chunks ──────────────────────────────────────────────────────
    header("PHASE 7 — Sample Chunks")

    import random
    random.seed(42)
    sample_indices = [0, 1] + random.sample(
        range(2, len(docs)), min(3, len(docs) - 2)
    )

    for idx in sample_indices:
        doc = docs[idx]
        m = doc.metadata
        print(f"\n{C.BOLD}  ── Chunk #{idx} ──────────────────────────────────{C.RESET}")
        print(f"  ID            : {m.get('chunk_id')}")
        print(f"  Section       : [{m.get('section')}] {m.get('section_title')}")
        print(f"  Parent        : {m.get('parent_section', '(root)')}")
        print(f"  Pages         : {m.get('page_start')} – {m.get('page_end')}")
        print(f"  Tokens (est.) : {m.get('approx_tokens')}")
        print(f"  Text preview  :")
        preview = doc.page_content[:400].replace("\n", "\n    ")
        print(f"    {preview}")
        print(f"    {'...' if len(doc.page_content) > 400 else ''}")

    # ── Summary ───────────────────────────────────────────────────────────────
    header("SUMMARY")

    print(f"""
  PDF pages     : {extraction.total_pages}
  Sections      : {len(sections)}
  Chunks        : {len(docs)}
  Issues        : {len(missing_meta)} missing metadata
                  {len(tiny_chunks)} tiny chunks
                  {len(huge_chunks)} oversized chunks
    """)

    if not missing_meta:
        ok("Parser output looks GOOD. Ready to proceed to Phase 3 (embeddings).")
        if tiny_chunks or huge_chunks:
            warn("Minor size issues found - review if needed, but not blocking.")
        sys.exit(0)
    else:
        error(f"{len(missing_meta)} CRITICAL issues (missing metadata). Fix before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    run_inspection()
