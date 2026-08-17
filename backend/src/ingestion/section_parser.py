"""
section_parser.py
=================
Structure-aware section detection for 3GPP TS 23.501.

KEY INSIGHT (from PDF analysis of actual pages):
-------------------------------------------------
3GPP TS 23.501 uses a TWO-LINE heading format:

    <section_number>     ← section number ALONE on a line
    <section_title>      ← title on the NEXT line (after optional blank)

For level-2+ sections this works perfectly:
    Line N:   "4.2.2"
    Line N+1: "Network Functions and entities"

For top-level (level-1) sections the format is DIFFERENT:
    - The section NUMBER appears in the running page header (which we strip).
    - The TITLE of the top-level section appears at the START of the page
      as a stand-alone line, followed immediately by the first subsection.

    Example from page 41:
      Line 0: "Architecture model and concepts"   ← Section 4 title
      Line 1: "4.1"                               ← First subsection number
      Line 2: "General concepts"                  ← Subsection 4.1 title

Strategy for level-1 sections:
    Detect them by recognising that a line at the top of a page is a title
    that PRECEDES a level-2 subsection (X.Y) and is NOT itself preceded by
    a section number → treat it as a level-1 section whose number is X.

Section numbering:
    Level 1:  4, 5, 6, 7, …   (top-level clauses)
    Level 2:  4.1, 4.2, …
    Level 3:  4.2.1, 4.2.2, …
    Level 4:  4.2.2.1, …
    Level 5:  4.2.2.1.1  (rare)

Annex sections: Annex A, Annex B, Annex A.1, …
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.ingestion.parser import RawPage

logger = logging.getLogger(__name__)


# ── Regex patterns ────────────────────────────────────────────────────────────

# A line that contains ONLY a section number (level 2+)
_SECTION_NUM_ONLY_RE = re.compile(
    r"^(\d{1,2}(?:\.\d+){1,5})$"      # requires at least one dot → level 2+
)

# Annex number line: "Annex A", "Annex B.1", etc.
_ANNEX_NUM_ONLY_RE = re.compile(
    r"^(Annex\s+[A-Z](?:\.\d+)*)$",
    re.IGNORECASE,
)

# A level-1 section number (single integer, 1–20)
_LEVEL1_NUM_RE = re.compile(r"^(\d{1,2})$")

# Lines that are definitely NOT section titles
_SKIP_TITLE_PATTERNS = [
    re.compile(r"^\d"),              # starts with digit
    re.compile(r"^-\s"),            # bullet
    re.compile(r"^NOTE"),           # NOTE
    re.compile(r"^Table\s"),        # table caption
    re.compile(r"^Figure\s"),       # figure caption
    re.compile(r"\.{5,}"),          # TOC dots
    re.compile(r"\s{3,}\d+\s*$"),  # TOC entry with page number
]

# Known top-level section titles that appear at page starts in TS 23.501
# We use these for reverse lookup: title → section number
_KNOWN_LEVEL1_TITLES: Dict[str, str] = {
    "Scope": "1",
    "References": "2",
    "Definitions, symbols and abbreviations": "3",
    "Architecture model and concepts": "4",
    "Procedures": "5",
    "Network function (NF) description": "6",
    "Identifiers and addressing": "7",
    "Security": "8",
    "Charging aspects": "9",
    "Network management aspects": "10",
    "Annex A": "Annex A",
    "Annex B": "Annex B",
}

# Minimum page after which we consider headings to be body (not TOC)
# The TOC of TS 23.501 V18 spans roughly pages 1–26
_TOC_END_PAGE = 26


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Section:
    """A single section in the 3GPP document tree."""
    number: str           # e.g. "4.2.2"
    title: str            # e.g. "Network Functions and entities"
    level: int            # 1=top, 2=sub, 3=sub-sub, …
    page_start: int
    page_end: int
    text: str             # body text (no heading line)
    parent_number: str    # parent section number ("" if root)
    children: List["Section"] = field(default_factory=list)

    @property
    def full_heading(self) -> str:
        return f"{self.number} {self.title}"

    @property
    def is_annex(self) -> bool:
        return self.number.upper().startswith("ANNEX")

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0

    def summary(self) -> str:
        return (
            f"[{self.number}] {self.title} "
            f"(pages {self.page_start}-{self.page_end}, "
            f"{self.word_count} words, {len(self.children)} children)"
        )


# ── Parser ────────────────────────────────────────────────────────────────────

class SectionParser:
    """
    Identifies section headings in 3GPP TS 23.501 PDF text.

    Two-phase detection:
    1. Level-2+ headings: detected via two-line pattern
       (section number alone, followed by title line).
    2. Level-1 headings: detected by reverse-lookup of known titles
       at page boundaries.

    Deduplication: The PDF Table of Contents lists every section
    number before the body.  We keep the LAST occurrence of each
    section number (the body entry, not the TOC entry).
    """

    def __init__(self, pages: List[RawPage]):
        self.pages = pages
        self._sections: List[Section] = []
        self._section_map: Dict[str, Section] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self) -> List[Section]:
        """Full parse → flat ordered list of Sections."""
        logger.info("Starting section parsing ...")

        annotated = self._build_annotated_lines()
        logger.info(f"  Total annotated lines: {len(annotated)}")

        # Detect level-2+ headings (two-line pattern)
        candidates = self._detect_level2plus(annotated)
        logger.info(f"  Level-2+ raw candidates: {len(candidates)}")

        # Detect level-1 headings (known title lookup)
        level1 = self._detect_level1(annotated)
        logger.info(f"  Level-1 raw candidates: {len(level1)}")

        # Detect Annex headings
        annexes = self._detect_annexes(annotated)
        logger.info(f"  Annex raw candidates: {len(annexes)}")

        # Merge all candidates
        all_candidates = candidates + level1 + annexes
        all_candidates.sort(key=lambda x: x[0])   # sort by line index

        # Deduplicate: keep last occurrence of each section number
        headings = self._deduplicate(all_candidates)
        logger.info(f"  After deduplication: {len(headings)} headings.")

        if not headings:
            logger.warning("No headings detected after deduplication!")
            return []

        # Post-filter: remove sections where page_start is inside the TOC
        # (these are stray TOC entries that weren't deduplicated correctly)
        headings = self._filter_toc_stragglers(headings, annotated)
        logger.info(f"  After TOC filter: {len(headings)} headings.")

        # Build sections from heading spans
        sections = self._build_sections(annotated, headings)
        logger.info(f"  Built {len(sections)} sections.")

        # Link parent-child hierarchy
        self._link_hierarchy(sections)

        self._sections = sections
        self._section_map = {s.number: s for s in sections}
        return sections

    def get_section(self, number: str) -> Optional[Section]:
        return self._section_map.get(number)

    def get_all_sections(self) -> List[Section]:
        return self._sections

    def print_tree(self, max_depth: int = 3) -> None:
        roots = [s for s in self._sections if s.level == 1]
        for root in roots:
            self._print_node(root, 0, max_depth)

    # ── Private: annotated lines ──────────────────────────────────────────────

    def _build_annotated_lines(self) -> List[Tuple[str, int]]:
        result: List[Tuple[str, int]] = []
        for page in self.pages:
            for line in page.text.split("\n"):
                result.append((line, page.page_number))
        return result

    # ── Private: level-2+ detection ───────────────────────────────────────────

    def _detect_level2plus(
        self, annotated: List[Tuple[str, int]]
    ) -> List[Tuple[int, str, str, int]]:
        """
        Detect level-2+ sections using the two-line pattern:
            line N   → section number (e.g. "4.2.2")
            line N+k → section title  (e.g. "Network Functions and entities")
        """
        candidates: List[Tuple[int, str, str, int]] = []
        n = len(annotated)

        for i, (line, page_num) in enumerate(annotated):
            stripped = line.strip()

            m = _SECTION_NUM_ONLY_RE.match(stripped)
            if not m:
                continue

            sec_num = m.group(1)
            # Skip section numbers that look like page numbers or table refs
            # (A valid section number must have a dot: checked by regex)

            # Look ahead for title
            title = self._find_title_ahead(annotated, i + 1, max_lookahead=4)
            if title:
                candidates.append((i, sec_num, title, page_num))

        return candidates

    # ── Private: TOC straggler filter ─────────────────────────────────────────

    def _filter_toc_stragglers(
        self,
        headings: List[Tuple[int, str, str, int]],
        annotated: List[Tuple[str, int]],
    ) -> List[Tuple[int, str, str, int]]:
        """
        Remove headings that appear to be TOC entries:
        - page_start < _TOC_END_PAGE (heading detected in TOC pages)
        - section number is unrealistically high (>20) for TS 23.501
        - section depth mismatch (e.g. a section with page_start < 30 but
          section number that should be deep in the document)
        """
        filtered: List[Tuple[int, str, str, int]] = []
        for item in headings:
            line_idx, sec_num, sec_title, page_num = item

            # Remove if page is in TOC zone
            if page_num < _TOC_END_PAGE:
                logger.debug(f"  Filtering TOC straggler: [{sec_num}] on page {page_num}")
                continue

            # Remove unrealistically large top-level section numbers
            # TS 23.501 has at most ~20 top-level sections
            top_num = sec_num.split(".")[0]
            if top_num.isdigit() and int(top_num) > 25:
                logger.debug(f"  Filtering unlikely section number: [{sec_num}]")
                continue

            filtered.append(item)

        return filtered

    # ── Private: level-1 detection ────────────────────────────────────────────

    def _detect_level1(
        self, annotated: List[Tuple[str, int]]
    ) -> List[Tuple[int, str, str, int]]:
        """
        Detect level-1 sections by finding known titles in the document body.

        In TS 23.501, top-level section titles appear:
        - At the start of a page (line 0 or 1), after the TOC
        - Followed shortly by a level-2 section number (e.g. "4.1")

        We use _KNOWN_LEVEL1_TITLES for reverse lookup.
        We also use a heuristic: find any line after the TOC that is
        immediately followed by a "X.Y" section number line, and that
        line matches a known level-1 title.
        """
        candidates: List[Tuple[int, str, str, int]] = []
        n = len(annotated)

        for i, (line, page_num) in enumerate(annotated):
            # Only look after TOC
            if page_num < _TOC_END_PAGE:
                continue

            stripped = line.strip()
            if not stripped:
                continue

            # Check if this line is a known level-1 title
            if stripped in _KNOWN_LEVEL1_TITLES:
                sec_num = _KNOWN_LEVEL1_TITLES[stripped]
                # Verify: next non-blank line should be a level-2 subsection
                # of this section
                confirmed = self._confirm_level1(annotated, i, sec_num)
                if confirmed:
                    candidates.append((i, sec_num, stripped, page_num))

        return candidates

    def _confirm_level1(
        self,
        annotated: List[Tuple[str, int]],
        title_idx: int,
        sec_num: str,
    ) -> bool:
        """
        Confirm a level-1 section by checking that the next section
        number found after this title starts with sec_num + ".".
        """
        n = len(annotated)
        for j in range(title_idx + 1, min(title_idx + 10, n)):
            line, _ = annotated[j]
            stripped = line.strip()
            if not stripped:
                continue
            m = _SECTION_NUM_ONLY_RE.match(stripped)
            if m:
                next_sec = m.group(1)
                # Should be a child of our section
                if next_sec.startswith(sec_num + "."):
                    return True
                # If a completely unrelated section, return False
                return False
        return False

    # ── Private: annex detection ──────────────────────────────────────────────

    def _detect_annexes(
        self, annotated: List[Tuple[str, int]]
    ) -> List[Tuple[int, str, str, int]]:
        """Detect Annex headings using the Annex-specific pattern."""
        candidates: List[Tuple[int, str, str, int]] = []
        n = len(annotated)

        for i, (line, page_num) in enumerate(annotated):
            stripped = line.strip()
            ann_m = _ANNEX_NUM_ONLY_RE.match(stripped)
            if not ann_m:
                continue

            sec_num = ann_m.group(1).strip()
            title = self._find_title_ahead(annotated, i + 1, max_lookahead=5)
            if not title:
                # Use generic title from the annex number
                title = f"{sec_num} content"
            candidates.append((i, sec_num, title, page_num))

        return candidates

    # ── Private: title lookahead ──────────────────────────────────────────────

    def _find_title_ahead(
        self,
        annotated: List[Tuple[str, int]],
        start: int,
        max_lookahead: int = 4,
    ) -> Optional[str]:
        """
        Look ahead from `start` for a valid title line.
        Skip blank lines. Return first valid title or None.
        """
        n = len(annotated)
        blanks = 0
        for j in range(start, min(start + max_lookahead + 3, n)):
            line, _ = annotated[j]
            stripped = line.strip()
            if not stripped:
                blanks += 1
                if blanks > 1:
                    return None
                continue
            if self._is_valid_title(stripped):
                return stripped
            # Hit another section number or non-title → stop
            return None
        return None

    @staticmethod
    def _is_valid_title(text: str) -> bool:
        """Return True if text looks like a section title."""
        if len(text) < 4 or len(text) > 200:
            return False
        # Must start with uppercase or "("
        if not (text[0].isupper() or text[0] == "("):
            return False
        # Must not match skip patterns
        for pattern in _SKIP_TITLE_PATTERNS:
            if pattern.search(text):
                return False
        # Must not be all-caps short (abbreviation line)
        if len(text) < 8 and text.isupper():
            return False
        return True

    # ── Private: deduplication ────────────────────────────────────────────────

    def _deduplicate(
        self, candidates: List[Tuple[int, str, str, int]]
    ) -> List[Tuple[int, str, str, int]]:
        """
        Keep the LAST occurrence of each section number.
        TOC entries appear first; body entries appear last.
        Result is sorted by line_index.
        """
        last: Dict[str, Tuple[int, str, str, int]] = {}
        for item in candidates:
            _, sec_num, _, _ = item
            last[sec_num] = item
        unique = list(last.values())
        unique.sort(key=lambda x: x[0])
        return unique

    # ── Private: build sections ───────────────────────────────────────────────

    def _build_sections(
        self,
        annotated: List[Tuple[str, int]],
        headings: List[Tuple[int, str, str, int]],
    ) -> List[Section]:
        """Slice text between consecutive headings → Section objects."""
        sections: List[Section] = []

        for i, (line_idx, sec_num, sec_title, page_start) in enumerate(headings):
            # Determine where body starts (after the title line)
            body_start = self._find_body_start(annotated, line_idx, sec_title)

            # Body ends at the next heading (or end of document)
            body_end = headings[i + 1][0] if i + 1 < len(headings) else len(annotated)

            # Extract body text
            body_lines = [annotated[j][0] for j in range(body_start, body_end)]
            body_text = "\n".join(body_lines).strip()
            body_text = re.sub(r"\n{3,}", "\n\n", body_text)

            # Page end = last page in body
            page_end = page_start
            if body_end > body_start:
                page_end = annotated[min(body_end - 1, len(annotated) - 1)][1]

            level = self._compute_level(sec_num)
            parent = self._compute_parent(sec_num)

            sections.append(Section(
                number=sec_num,
                title=sec_title,
                level=level,
                page_start=page_start,
                page_end=page_end,
                text=body_text,
                parent_number=parent,
            ))

        return sections

    def _find_body_start(
        self,
        annotated: List[Tuple[str, int]],
        heading_idx: int,
        title: str,
    ) -> int:
        """Find the line index right after the title line."""
        n = len(annotated)
        for j in range(heading_idx + 1, min(heading_idx + 6, n)):
            line, _ = annotated[j]
            if line.strip() == title.strip():
                return j + 1
        return heading_idx + 1

    # ── Private: hierarchy ────────────────────────────────────────────────────

    def _link_hierarchy(self, sections: List[Section]) -> None:
        num_map: Dict[str, Section] = {s.number: s for s in sections}
        for s in sections:
            if s.parent_number:
                parent = num_map.get(s.parent_number)
                if parent:
                    parent.children.append(s)

    @staticmethod
    def _compute_level(sec_num: str) -> int:
        if sec_num.upper().startswith("ANNEX"):
            parts = sec_num.split()
            return len(parts[1].split(".")) if len(parts) > 1 else 1
        return len(sec_num.split("."))

    @staticmethod
    def _compute_parent(sec_num: str) -> str:
        if sec_num.upper().startswith("ANNEX"):
            parts = sec_num.split()
            if len(parts) > 1:
                sub = parts[1].split(".")
                return "Annex " + ".".join(sub[:-1]) if len(sub) > 1 else ""
            return ""
        parts = sec_num.split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else ""

    def _print_node(self, section: Section, depth: int, max_depth: int) -> None:
        indent = "  " * depth
        print(f"{indent}{section.summary()}")
        if depth < max_depth:
            for child in section.children[:15]:
                self._print_node(child, depth + 1, max_depth)
