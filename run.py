"""
run.py
======
Simple Terminal Chatbot for 3GPP 5G Standards Assistant.
Usage:
    python run.py
"""

import sys
import io
import os
import time
import warnings
from pathlib import Path

# Suppress deprecation and verbosity warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Force UTF-8 encoding for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR / "backend") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "backend"))

from backend.src.pipeline import RAGPipeline


def main():
    print("=" * 65)
    print(" 📡 3GPP 5G Standards Assistant Terminal Chatbot")
    print(" Specification: 3GPP TS 23.501 V18.12.0 (Release 18)")
    print(" Type your technical question below. Type 'exit' or 'q' to quit.")
    print("=" * 65)
    
    print("\nLoading 3GPP RAG Engine...")
    try:
        pipeline = RAGPipeline()
        print("Engine Ready! Ask your question.\n")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    while True:
        try:
            print("\nYou: ", end="", flush=True)
            user_input = sys.stdin.readline()
            if not user_input:
                break
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Chatbot. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nExiting Chatbot. Goodbye!")
            break

        print("\nThinking & searching 3GPP TS 23.501 specification...")
        start_time = time.time()
        result = pipeline.query(user_input, debug=False)
        elapsed = time.time() - start_time

        print("\nAssistant:")
        if result.abstained:
            print(f"[Abstained] {result.answer}")
            if result.abstain_reason:
                print(f"Reason: {result.abstain_reason}")
        else:
            print(result.answer)
            
            if result.sources:
                print(f"\n[Verified Citations ({elapsed:.2f}s)]:")
                for i, src in enumerate(result.sources[:5], start=1):
                    pages = f"pp. {src.page_start}-{src.page_end}" if src.page_start != src.page_end else f"p. {src.page_start}"
                    print(f"  ({i}) Section {src.section} ({src.section_title}) | {pages}")


if __name__ == "__main__":
    main()
