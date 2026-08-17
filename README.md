# 3GPP 5G Standards Assistant 

A **Retrieval-Augmented Generation (RAG)** chatbot designed specifically for answering complex technical inquiries against **3GPP TS 23.501 V18.12.0** (*5G System Architecture, Release 18*).

Equipped with a **Multi-Query Hybrid Search Engine** (Dense + BM25 + Reciprocal Rank Fusion), **Cross-Encoder Reranking**, an **Evidence Sufficiency Gate**, an **Interactive Terminal Chatbot**, and **Swagger UI Documentation**.

---

## Key Technical Features

1. **Multi-Query Decomposition & Acronym-Aware Hybrid Search**:
   - **Sub-Aspect Decomposition**: Decomposes complex telecom inquiries into sub-queries for parallel retrieval.
   - **Dense Search**: `BAAI/bge-base-en-v1.5` embeddings stored in **ChromaDB**.
   - **Sparse BM25 Search**: Preserves 3GPP specific acronyms (`AMF`, `SMF`, `UPF`, `UCMF`, `5QI`, `PDU`, `S-NSSAI`).
   - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse ranks dynamically.

2. **Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`)**:
   - Scores retrieved candidates with cross-attention to filter out irrelevancies.

3. **Evidence Sufficiency Gate (Zero-Hallucination Guardrail)**:
   - Evaluates evidence quality *before* calling the LLM. If relevance score $< 0.15$, the engine abstains from answering rather than guessing.

4. **Section-Aware Ingestion Pipeline**:
   - Parses PyMuPDF pages while preserving 3GPP hierarchical section structures (`5.2.18.2`, `5.4.4.1`) and page numbers.

5. **Terminal Chatbot (`python run.py`) & Swagger UI**:
   - Run interactive queries directly in your terminal via `python run.py` or use FastAPI Swagger UI at `http://localhost:8000/docs`.

---

## System Architecture

```
                       3GPP TS 23.501 PDF (721 pages)
                                     │
                                     ▼
                      Section-Aware Ingestion Pipeline
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
           ChromaDB Vector Store                   BM25 Sparse Index
           (BGE-base-en-v1.5)                     (Acronym preservation)
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     ▼
                       Multi-Query Sub-Aspect Fusion
                                     │
                                     ▼
                         Reciprocal Rank Fusion (RRF)
                                     │
                                     ▼
                           Cross-Encoder Reranker
                        (ms-marco-MiniLM-L-6-v2)
                                     │
                                     ▼
                          Evidence Sufficiency Gate
                          /                       \
             Score < 0.15                         Score >= 0.15
                  │                                    │
                  ▼                                    ▼
            Refuse / Abstain                     LangChain LCEL
          (Zero Hallucination)              (Prompt | LLM | Parser)
                                                       │
                                                       ▼
                                            Top-5 Page & Section Citations
```

---

## Benchmark Evaluation Metrics

Evaluated across a 50-question benchmark dataset (35 in-scope technical questions, 15 out-of-scope / adversarial questions):

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Citation Accuracy** | **82.9%** | Ground-truth specification section retrieved and cited correctly |
| **Abstention Accuracy** | **100.0%** | 100% defense against out-of-scope and adversarial prompts |
| **Groundedness** | **94.3%** | Answers provided for valid in-scope technical questions |

---

## Quickstart & Execution

### 1. Prerequisites & Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Jas0108/3gpp-rag.git
cd 3gpp-rag
pip install -r backend/requirements.txt
```

### 2. Configure Environment Variables
Copy `backend/.env.example` to `backend/.env` and set your API key:
```env
LLM_PROVIDER=openrouter
LLM_MODEL=google/gemma-4-26b-a4b-it:free
LLM_API_KEY=your_openrouter_api_key_here
```

### 3. Launch Terminal Chatbot (Recommended)
Launch the interactive terminal chatbot:
```bash
python run.py
```

### 4. Launch FastAPI & Swagger UI (Optional)
To launch the FastAPI REST service and interactive Swagger docs:
```bash
uvicorn backend.api.main:app --reload --port 8000
```
Open **`http://localhost:8000/docs`** in your browser.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
