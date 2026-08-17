"""
cli.py
======
Interactive Terminal Interface for 3GPP 5G Standards Assistant.
Run: python cli.py
"""

import sys
import time
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR / "backend") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "backend"))

from backend.src.pipeline import RAGPipeline


def print_banner():
    print("\n" + "=" * 70)
    print(" 3GPP 5G STANDARDS ASSISTANT -- INTERACTIVE TERMINAL UI")
    print(" 3GPP TS 23.501 V18.12.0 (5G System Architecture, Release 18)")
    print("=" * 70)


def print_sample_questions():
    print("\nSuggested Test Queries:")
    print("  [1] Tell me about UE radio Capability Management Function (UCMF)")
    print("  [2] What are the Principles for Binding, Selection and Reselection?")
    print("  [3] Tell me about Maximum Packet Loss Rate")
    print("  [4] What are Dual Connectivity based end to end Redundant User Plane Paths?")
    print("  [5] What is UE's Mobility Event Management?")
    print("  [6] Tell me about MBSR authorization")
    print("  [exit / q] Quit application")
    print("-" * 70)


def main():
    print_banner()
    print("\nLoading 3GPP RAG Engine (ChromaDB + BM25 + BGE Embeddings)...")
    
    try:
        pipeline = RAGPipeline()
        print("Pipeline Initialized Successfully!\n")
    except Exception as e:
        print(f"\nError initializing pipeline: {e}")
        return

    sample_map = {
        "1": "Tell me about UE radio Capability Management Function (UCMF)",
        "2": "What are the Principles for Binding, Selection and Reselection?",
        "3": "Tell me about Maximum Packet Loss Rate",
        "4": "What are Dual Connectivity based end to end Redundant User Plane Paths?",
        "5": "What is UE's Mobility Event Management?",
        "6": "Tell me about MBSR authorization",
    }

    while True:
        print_sample_questions()
        try:
            user_input = input("Enter option number (1-6) or type your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nExiting 3GPP Assistant. Goodbye!\n")
            break

        query = sample_map.get(user_input, user_input)

        print("\n" + "-" * 70)
        print(f"QUERY: '{query}'")
        print("Searching 3GPP TS 23.501 specification & generating answer...")
        print("-" * 70)

        start_time = time.time()
        result = pipeline.query(query, debug=False)
        elapsed = time.time() - start_time

        print(f"\nResponse Time: {elapsed:.2f}s")

        if result.abstained:
            print("\nEVIDENCE GATE TRIGGERED (ABSTAINED)")
            print(f"Answer: {result.answer}")
            if result.abstain_reason:
                print(f"Reason: {result.abstain_reason}")
        else:
            print("\nGROUNDED ANSWER:")
            print(result.answer)

            if result.sources:
                print("\nVERIFIED SOURCE CITATIONS:")
                for i, src in enumerate(result.sources[:5], start=1):
                    pages = f"pp. {src.page_start}-{src.page_end}" if src.page_start != src.page_end else f"p. {src.page_start}"
                    sub = f" | Sub-sections: {src.all_sections}" if src.all_sections else ""
                    print(f"   [{i}] Section {src.section} ({src.section_title}) | {pages}{sub}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
