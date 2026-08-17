# 3GPP 5G Standards Assistant 

A **Retrieval-Augmented Generation (RAG)** system designed specifically for answering complex technical inquiries against **3GPP TS 23.501 V18.12.0** (*5G System Architecture, Release 18*).

Equipped with a **Multi-Query Hybrid Search Engine** (Dense + BM25 + Reciprocal Rank Fusion), **Cross-Encoder Reranking**, an **Evidence Sufficiency Gate**, and dual deployment interfaces (**Streamlit Web App** & **FastAPI REST Service**).

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

5. **Flexible Web UI & API Interface**:
   - Includes both a Streamlit Cloud Web Application (`streamlit_app.py`) and a centered FastAPI web interface (`frontend/`).

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

## Repository Structure

```
3gpp-rag-chatbot/
├── backend/                      # Python Backend Service
│   ├── api/
│   │   └── main.py               # FastAPI REST API & static server
│   ├── data/
│   │   └── 23501-ic0.pdf         # 3GPP Specification corpus
│   ├── evaluation/
│   │   ├── questions.json        # 50-question benchmark dataset
│   │   └── results.json          # Standardized metric outputs
│   ├── src/
│   │   ├── config.py             # Global settings & environment loader
│   │   ├── schemas.py            # Pydantic v2 data models
│   │   ├── pipeline.py           # End-to-end RAG orchestrator
│   │   ├── ingestion/            # PyMuPDF parser & section chunker
│   │   ├── retrieval/            # Hybrid multi-query retriever
│   │   ├── generation/           # Evidence gate & LangChain generator
│   │   └── evaluation/           # Benchmark evaluator tool
│   ├── .env.example              # Environment key template
│   └── requirements.txt          # Backend dependencies
│
├── frontend/                     # Fast HTML/CSS Web UI
│   ├── index.html                # Centered, minimalistic layout
│   ├── style.css                 # Dark glassmorphism styling
│   └── app.js                    # Query handling & citation rendering
│
├── streamlit_app.py              # Streamlit Community Cloud Application
├── .gitignore                    # Environment & index exclusions
└── README.md
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

## Quickstart & Setup

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
LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
LLM_API_KEY=your_openrouter_api_key_here
```

### 3. Run Ingestion (Indexes PDF Corpus)
Run PDF extraction and build ChromaDB + BM25 indices:
```bash
python -m backend.src.ingestion.ingest
```

### 4. Launch Option A: Streamlit Web Application
Run locally:
```bash
streamlit run streamlit_app.py
```

### 5. Launch Option B: FastAPI Server & Web UI
Navigate into `backend` and launch the FastAPI server:
```bash
cd backend
uvicorn api.main:app --reload --port 8000
```
- **Web Interface**: Open **`http://localhost:8000`**
- **Interactive API Docs**: **`http://localhost:8000/docs`**

---

## 1-Click Cloud Deployment (Streamlit Community Cloud)

To deploy a live hosted version for interviewers (100% Free - 1 GB RAM):
1. Push your repository to GitHub.
2. Sign in at **share.streamlit.io**.
3. Create App -> Select `Jas0108/3gpp-rag` -> Main file path: `streamlit_app.py`.
4. In **Advanced settings... Secrets**, add:
   ```toml
   LLM_PROVIDER = "openrouter"
   LLM_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
   LLM_API_KEY = "your_actual_api_key"
   ```

---

## API Reference

### `POST /api/chat`
Process a technical question and return a grounded answer with top-5 section and page citations.

**Request**:
```json
{
  "question": "Tell me about UE radio Capability Management Function (UCMF)"
}
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.
