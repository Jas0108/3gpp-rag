/* =============================================================================
   3GPP RAG Chatbot — Minimalistic Client Application Logic
   ============================================================================= */

const API_BASE_URL = "http://localhost:8000";

// DOM Elements
const statusBadge = document.getElementById("system-status");
const statusText = document.getElementById("status-text");
const questionInput = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const responseContainer = document.getElementById("response-container");
const loadingState = document.getElementById("loading-state");
const outputCard = document.getElementById("output-card");
const responseIcon = document.getElementById("response-icon");
const responseStatusTitle = document.getElementById("response-status-title");
const elapsedTime = document.getElementById("elapsed-time");
const answerBody = document.getElementById("answer-body");
const citationsSection = document.getElementById("citations-section");
const citationsList = document.getElementById("citations-list");
const copyBtn = document.getElementById("copy-btn");
const promptChips = document.querySelectorAll(".prompt-chip");

// Check Backend Health on Load
async function checkHealth() {
  if (!statusBadge || !statusText) return;
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      if (data.status === "ok" || data.status === "degraded") {
        statusBadge.className = "status-badge online";
        statusText.textContent = "Backend Online • TS 23.501 V18.12.0";
      } else {
        statusBadge.className = "status-badge offline";
        statusText.textContent = "Backend Degraded";
      }
    } else {
      throw new Error("HTTP error");
    }
  } catch (err) {
    statusBadge.className = "status-badge offline";
    statusText.textContent = "Backend Offline (Start FastAPI at :8000)";
  }
}

// Process Question Query
async function handleQuery(queryText) {
  const query = queryText || questionInput.value.trim();
  if (!query) return;

  // Update UI for loading state
  sendBtn.disabled = true;
  responseContainer.classList.remove("hidden");
  loadingState.classList.remove("hidden");
  outputCard.classList.add("hidden");

  const startTime = performance.now();

  try {
    const res = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question: query, debug: false }),
    });

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
    loadingState.classList.add("hidden");
    outputCard.classList.remove("hidden");

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Server error occurred");
    }

    const data = await res.json();
    renderResponse(data, elapsed);
  } catch (err) {
    loadingState.classList.add("hidden");
    outputCard.classList.remove("hidden");
    outputCard.className = "output-card abstained";
    responseIcon.textContent = "⚠️";
    responseStatusTitle.textContent = "Error";
    elapsedTime.textContent = "⚡ 0.0s";
    answerBody.textContent = `Unable to process request: ${err.message}. Make sure the FastAPI server is running on http://localhost:8000.`;
    citationsSection.classList.add("hidden");
  } finally {
    sendBtn.disabled = false;
  }
}

// Render Response Data
function renderResponse(data, elapsedSeconds) {
  elapsedTime.textContent = `⚡ ${elapsedSeconds}s`;

  if (data.abstained) {
    outputCard.className = "output-card abstained";
    responseIcon.textContent = "🛡️";
    responseStatusTitle.textContent = "Evidence Gate Triggered (Abstained)";
    answerBody.textContent = data.answer;
    citationsSection.classList.add("hidden");
  } else {
    outputCard.className = "output-card";
    responseIcon.textContent = "💡";
    responseStatusTitle.textContent = "Grounded Answer";
    answerBody.textContent = data.answer;

    // Render Sources / Citations (Top 5 Only)
    if (data.sources && data.sources.length > 0) {
      citationsSection.classList.remove("hidden");
      citationsList.innerHTML = data.sources
        .slice(0, 5)
        .map(
          (src) => {
            const pageStr = (src.page_start && src.page_end && src.page_start !== src.page_end)
              ? `Pages ${src.page_start}–${src.page_end}`
              : `Page ${src.page_start || src.page_end || "N/A"}`;
            return `
        <div class="citation-item">
          <div class="citation-header">
            <span>Section ${src.section || "N/A"}</span>
            <span>${pageStr}</span>
          </div>
          ${src.all_sections ? `<div class="citation-snippet">Sub-sections: ${escapeHtml(src.all_sections)}</div>` : ''}
        </div>
      `;
          }
        )
        .join("");
    } else {
      citationsSection.classList.add("hidden");
    }
  }
}

// Helper to escape HTML characters
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Event Listeners
sendBtn.addEventListener("click", () => handleQuery());

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleQuery();
  }
});

promptChips.forEach((chip) => {
  chip.addEventListener("click", () => {
    const query = chip.getAttribute("data-query");
    questionInput.value = query;
    handleQuery(query);
  });
});

copyBtn.addEventListener("click", () => {
  const textToCopy = answerBody.textContent;
  if (!textToCopy) return;
  navigator.clipboard.writeText(textToCopy).then(() => {
    const orig = copyBtn.textContent;
    copyBtn.textContent = "✅ Copied!";
    setTimeout(() => {
      copyBtn.textContent = orig;
    }, 2000);
  });
});

// Run health check on startup
checkHealth();
setInterval(checkHealth, 10000);
