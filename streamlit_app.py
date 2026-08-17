"""
streamlit_app.py
================
Streamlit Web Interface for 3GPP 5G Standards Assistant RAG Pipeline.
Deployable directly on Streamlit Community Cloud (1 GB Free RAM).
"""

import sys
from pathlib import Path
import time
import streamlit as st

# Add backend to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR / "backend") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "backend"))

from backend.src.pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="3GPP 5G Standards Assistant",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom Styling (Dark Glassmorphism aesthetic)
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    .citation-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        color: #38bdf8;
        font-size: 14px;
        margin-bottom: 4px;
    }
    .citation-sub {
        font-size: 12px;
        color: #94a3b8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing 3GPP RAG Engine & Ingesting PDF Specification...")
def get_pipeline():
    """Initializes and caches the RAG pipeline instance."""
    pipe = RAGPipeline()
    # If vector index is empty (fresh clone), run ingestion automatically
    if pipe.chunk_count() == 0:
        from backend.src.ingestion.ingest import run_ingestion
        run_ingestion()
        pipe = RAGPipeline()
    return pipe


# Application Header
st.title("📡 3GPP Standards Assistant")
st.caption("3GPP TS 23.501 V18.12.0 • 5G System Architecture")

# Initialize Pipeline
try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Failed to load RAG pipeline: {e}")
    st.stop()

st.markdown("---")

# User Query Input
default_val = st.session_state.get("user_query", "")
question = st.text_area(
    "Ask a technical question about 5G specifications:",
    value=default_val,
    placeholder="e.g. Tell me about UE radio Capability Management Function (UCMF)",
    height=100,
)

# Process Question Button
submit_clicked = st.button("Ask Assistant", type="primary", use_container_width=True)

st.markdown("---")

# Sample Prompts Section (Stacked vertically below text area)
st.caption("Try asking:")

if st.button("Tell me about UE radio Capability Management Function (UCMF)", use_container_width=True):
    st.session_state["user_query"] = "Tell me about UE radio Capability Management Function (UCMF)"
    st.session_state["auto_submit"] = True
    st.rerun()

if st.button("What are the Principles for Binding, Selection and Reselection?", use_container_width=True):
    st.session_state["user_query"] = "What are the Principles for Binding, Selection and Reselection?"
    st.session_state["auto_submit"] = True
    st.rerun()

if st.button("Tell me about Maximum Packet Loss Rate", use_container_width=True):
    st.session_state["user_query"] = "Tell me about Maximum Packet Loss Rate"
    st.session_state["auto_submit"] = True
    st.rerun()

# Determine if search should trigger
should_run = submit_clicked or st.session_state.get("auto_submit", False)
if st.session_state.get("auto_submit", False):
    st.session_state["auto_submit"] = False

# Execution & Results Section
if should_run:
    query_text = question.strip() or st.session_state.get("user_query", "").strip()
    
    if not query_text:
        st.warning("Please enter a question or click one of the sample queries.")
    else:
        st.markdown("---")
        
        # Prominent Instant Banner + Spinner
        status_box = st.empty()
        status_box.info("⏳ Searching 3GPP TS 23.501 specification for answer...")
        
        with st.spinner("Processing query..."):
            start_time = time.time()
            result = pipeline.query(query_text, debug=False)
            elapsed = time.time() - start_time
            
        status_box.empty()  # Clear status banner when done

        # Render Response
        if result.abstained:
            st.error(f"🛡️ **Evidence Gate Triggered (Abstained)**\n\n{result.answer}")
            if result.abstain_reason:
                st.caption(f"Reason: {result.abstain_reason}")
        else:
            st.success(f"💡 **Grounded Answer** *(processed in {elapsed:.2f}s)*")
            st.markdown(result.answer)

            # Render Sources (Top 5)
            if result.sources:
                st.markdown("### 📄 Verified Source Citations")
                for src in result.sources[:5]:
                    pages = (
                        f"Pages {src.page_start}–{src.page_end}"
                        if src.page_start != src.page_end
                        else f"Page {src.page_start}"
                    )
                    sec_display = f"Section {src.section}"
                    sub_display = f"Sub-sections: {src.all_sections}" if src.all_sections else ""

                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span>{sec_display}</span>
                                <span>{pages}</span>
                            </div>
                            <div class="citation-sub">{sub_display}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
