"""
evaluator.py
============
RAG pipeline evaluation metrics.

Metrics implemented:
- Citation Accuracy      : do cited sections match reference_sections?
- Abstention Accuracy    : did the system correctly abstain on out-of-scope/adversarial Qs?
- Groundedness           : are answers provided for valid in-scope Qs?

Run:
    python -m src.evaluation.evaluator
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Force UTF-8 output
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).resolve().parent.parent.parent / "evaluation" / "questions.json"
RESULTS_PATH   = Path(__file__).resolve().parent.parent.parent / "evaluation" / "results.json"


# ── Metric helpers ────────────────────────────────────────────────────────────

def section_matches(retrieved: str, reference: str) -> bool:
    """
    Returns True if retrieved section matches reference section.
    Supports exact, parent-child (e.g., 6.2 matching 6.2.1), and section branch matches.
    """
    r = retrieved.strip().rstrip(".")
    ref = reference.strip().rstrip(".")
    if r == ref:
        return True
    if r.startswith(ref + ".") or ref.startswith(r + "."):
        return True
    r_parts = r.split(".")
    ref_parts = ref.split(".")
    if len(r_parts) >= 2 and len(ref_parts) >= 2 and r_parts[:2] == ref_parts[:2]:
        return True
    return False


def citation_correct(
    cited_sections: List[str],
    reference_sections: List[str],
) -> bool:
    """Return True if at least one cited section matches a reference section."""
    if not reference_sections:
        return True
    return any(
        section_matches(ret, ref)
        for ret in cited_sections
        for ref in reference_sections
    )


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs the evaluation dataset through the RAG pipeline and computes metrics.
    """

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def run(
        self, questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run all questions and compute aggregate metrics.

        Args:
            questions: List of question dicts from questions.json.

        Returns:
            Dict of metric results.
        """
        results = []
        n_total = len(questions)
        n_should_answer = sum(
            1 for q in questions if q.get("expected_behavior") == "answer"
        )
        n_should_abstain = sum(
            1 for q in questions if q.get("expected_behavior") == "abstain"
        )

        # Counters
        citation_correct_count = 0
        abstain_correct = 0
        answer_correct = 0
        total_answered = 0

        logger.info(
            f"Running evaluation on {n_total} questions "
            f"({n_should_answer} expected answers, "
            f"{n_should_abstain} expected abstentions) ..."
        )

        for i, q in enumerate(questions, start=1):
            qid = q.get("id", f"Q{i}")
            question = q.get("question", "")
            expected = q.get("expected_behavior", "answer")
            ref_sections = q.get("reference_sections", [])

            logger.info(f"  [{i}/{n_total}] {qid}: {question[:60]}...")

            t0 = time.time()
            try:
                response = self.pipeline.query(question, debug=False)
            except Exception as e:
                logger.error(f"    Pipeline error: {e}")
                results.append({
                    "id": qid,
                    "question": question,
                    "expected": expected,
                    "abstained": True,
                    "error": str(e),
                })
                continue
            elapsed = round(time.time() - t0, 2)

            abstained = response.abstained
            cited_sections = []
            for s in response.sources:
                if s.all_sections:
                    cited_sections.extend([sec.strip() for sec in s.all_sections.split(",") if sec.strip()])
                else:
                    cited_sections.append(s.section)
            cited_sections = list(dict.fromkeys(cited_sections))

            if expected == "answer" and ref_sections:
                correct_cite = citation_correct(cited_sections, ref_sections)
                if correct_cite:
                    citation_correct_count += 1

            # Abstention accuracy
            if expected == "abstain":
                if abstained:
                    abstain_correct += 1
            else:
                total_answered += 1
                if not abstained:
                    answer_correct += 1

            result_item = {
                "id": qid,
                "question": question,
                "category": q.get("category", ""),
                "expected": expected,
                "abstained": abstained,
                "answer_preview": response.answer[:200],
                "cited_sections": cited_sections,
                "reference_sections": ref_sections,
                "elapsed_s": elapsed,
            }
            if ref_sections:
                result_item["citation_correct"] = citation_correct(cited_sections, ref_sections)

            results.append(result_item)
            logger.info(
                f"    abstained={abstained} | "
                f"cited={cited_sections[:3]} | {elapsed}s"
            )

        # ── Aggregate metrics ──────────────────────────────────────────────────
        abstention_accuracy = (
            abstain_correct / n_should_abstain if n_should_abstain else 1.0
        )
        groundedness = answer_correct / total_answered if total_answered else 0.0
        citation_accuracy = (
            citation_correct_count / n_should_answer if n_should_answer else 0.0
        )

        metrics = {
            "total_questions": n_total,
            "questions_should_answer": n_should_answer,
            "questions_should_abstain": n_should_abstain,
            "citation_accuracy": round(citation_accuracy, 4),
            "abstention_accuracy": round(abstention_accuracy, 4),
            "groundedness": round(groundedness, 4),
        }

        return {
            "metrics": metrics,
            "per_question": results,
        }


def main():
    """CLI entry point."""
    if not QUESTIONS_PATH.exists():
        logger.error(f"Questions file not found: {QUESTIONS_PATH}")
        sys.exit(1)

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    pipeline = RAGPipeline()

    ready = pipeline.is_ready()
    if not ready.get("chroma_ready") or not ready.get("bm25_ready"):
        logger.error(
            "Pipeline not ready. Run ingestion first: python -m src.ingestion.ingest"
        )
        sys.exit(1)

    evaluator = Evaluator(pipeline)
    output = evaluator.run(questions)

    # Print metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in output["metrics"].items():
        print(f"  {k:<35}: {v}")

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
