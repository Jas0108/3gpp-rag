# 3GPP 5G Standards Assistant 

A **Retrieval-Augmented Generation (RAG)** chatbot designed specifically for answering complex technical inquiries against **3GPP TS 23.501 V18.12.0** (*5G System Architecture, Release 18*).

Equipped with a **Multi-Query Hybrid Search Engine** (Dense + BM25 + Reciprocal Rank Fusion), **Cross-Encoder Reranking**, an **Evidence Sufficiency Gate**, and an **Interactive FastAPI / Swagger API Service**.

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

5. **FastAPI & Interactive Swagger UI**:
   - High-throughput REST API with interactive Swagger documentation (`/docs`) for testing inquiries directly in the browser or terminal via `curl`.

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

## Quickstart & Local FastAPI Execution

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

### 3. Launch FastAPI Backend
From the project root directory, launch the Uvicorn server:
```bash
uvicorn backend.api.main:app --reload --port 8000
```

---

## Interactive Swagger UI & Terminal Execution

### Option A: Interactive Swagger UI
Open your browser to: **`http://localhost:8000/docs`**
- Click **`POST /api/chat`** ➡️ **Try it out**
- Enter your question JSON payload and click **Execute**!

### Option B: Query via Terminal (`curl`)
Execute questions directly in your terminal:

```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d "{\"question\": \"Tell me about UE radio Capability Management Function (UCMF)\"}"
```

---

## API Reference

### `POST /api/chat`
Process a technical question and return a grounded answer with top-5 section and page citations.

**Request Payload**:
```json
{
  "question": "Tell me about UE radio Capability Management Function (UCMF)"
}
```

**Response Payload**:
```json
{
  "answer": "The UE radio Capability Management Function (UCMF) is used for the storage of dictionary entries...",
  "sources": [
    {
      "specification": "3GPP TS 23.501",
      "release": "18",
      "version": "V18.12.0",
      "section": "6.2.21",
      "section_title": "UE radio Capability Management Function (UCMF)",
      "page_start": 536,
      "page_end": 536
    }
  ],
  "abstained": false,
  "abstain_reason": null
}
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.
