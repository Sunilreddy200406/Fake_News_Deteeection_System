const newsText = document.getElementById("newsText");
const sourceUrl = document.getElementById("sourceUrl");
const checkBtn = document.getElementById("checkBtn");
const resultDiv = document.getElementById("result");

const historyBody = document.querySelector("#historyTable tbody");
const resultFilter = document.getElementById("resultFilter");
const applyFilterBtn = document.getElementById("applyFilterBtn");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const prevPageBtn = document.getElementById("prevPageBtn");
const nextPageBtn = document.getElementById("nextPageBtn");
const pageInfo = document.getElementById("pageInfo");

let currentPage = 1;
const pageSize = 10;
let currentFilter = "";

function showResult(message, cssClass) {
  resultDiv.className = `result ${cssClass}`;
  resultDiv.innerHTML = message;
}

function resultClass(value) {
  if (value === "Real") return "result-real";
  if (value === "Fake") return "result-fake";
  return "result-unverified";
}

function escapeHtml(text) {
  return (text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderHistory(items) {
  historyBody.innerHTML = "";
  if (!items.length) {
    historyBody.innerHTML = `<tr><td colspan="5">No history records found.</td></tr>`;
    return;
  }

  for (const row of items) {
    const source = row.source_url ? escapeHtml(row.source_url) : "-";
    historyBody.innerHTML += `
      <tr>
        <td>${escapeHtml(row.created_at)}</td>
        <td>${escapeHtml(row.news_summary)}</td>
        <td>${source}</td>
        <td><span class="${resultClass(row.result)}">${escapeHtml(row.result)}</span></td>
        <td>${escapeHtml(row.method)}</td>
      </tr>
    `;
  }
}

async function loadHistory(page = 1) {
  const params = new URLSearchParams({
    page: String(page),
    limit: String(pageSize),
  });
  if (currentFilter) params.set("result", currentFilter);

  const response = await fetch(`/history?${params.toString()}`);
  const data = await response.json();
  renderHistory(data.items || []);
  currentPage = data.page || page;
  pageInfo.textContent = `Page ${currentPage}`;
  prevPageBtn.disabled = currentPage <= 1;
  nextPageBtn.disabled = !(data.items && data.items.length === pageSize);
}

checkBtn.addEventListener("click", async () => {
  const text = newsText.value.trim();
  const url = sourceUrl.value.trim();
  if (!text) {
    showResult("Please enter some text.", "error");
    return;
  }

  showResult("Checking...", "error");

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, source_url: url }),
    });

    const data = await response.json();
    if (!response.ok) {
      showResult(data.error || "Prediction failed.", "error");
      return;
    }

    const finalLabel = data.final_label || data.prediction;
    const confidenceText = data.confidence
      ? `${(data.confidence * 100).toFixed(2)}%`
      : "N/A";
    const sourceDomain = data.source_domain || "Not provided";
    const sim = data.similarity ? data.similarity.best : null;
    const similarityText = sim !== null ? sim.toFixed(4) : "N/A";
    const keywords = Array.isArray(data.keywords) ? data.keywords.join(", ") : "";
    const entities = Array.isArray(data.entities) ? data.entities.join(", ") : "";
    const matchedArticle = data.matched_article
      ? `<div><strong>Matched Article:</strong> <a href="${data.matched_article.link}" target="_blank" rel="noopener noreferrer">${escapeHtml(data.matched_article.title)}</a></div>`
      : `<div><strong>Matched Article:</strong> Not found</div>`;

    let cssClass = "suspicious";
    if ((data.result || "").toLowerCase() === "real") cssClass = "verified";
    if ((data.result || "").toLowerCase() === "fake") cssClass = "fake";

    showResult(
      `<div><strong>Final Decision:</strong> ${escapeHtml(finalLabel)}</div>
       <div><strong>Result Category:</strong> <span class="${resultClass(data.result)}">${escapeHtml(data.result || "Unverified")}</span></div>
       <div><strong>Verification Method:</strong> ${escapeHtml(data.verification_method || "N/A")}</div>
       <div><strong>Reasoning:</strong> ${escapeHtml(data.reasoning || "N/A")}</div>
       <div><strong>ML Confidence:</strong> ${confidenceText}</div>
       <div><strong>Source Domain:</strong> ${escapeHtml(sourceDomain)}</div>
       <div><strong>Best Similarity Score:</strong> ${similarityText}</div>
       <div><strong>Decision Path:</strong> ${escapeHtml(data.decision_path || "N/A")}</div>
       ${matchedArticle}
       <div><strong>Keywords:</strong> ${escapeHtml(keywords || "N/A")}</div>
       <div><strong>Entities:</strong> ${escapeHtml(entities || "N/A")}</div>`,
      cssClass
    );

    // Always show latest entry immediately after each verification.
    currentFilter = "";
    resultFilter.value = "";
    await loadHistory(1);
  } catch (error) {
    showResult("Server error. Please try again.", "error");
  }
});

applyFilterBtn.addEventListener("click", async () => {
  currentFilter = resultFilter.value;
  await loadHistory(1);
});

prevPageBtn.addEventListener("click", async () => {
  if (currentPage > 1) {
    await loadHistory(currentPage - 1);
  }
});

nextPageBtn.addEventListener("click", async () => {
  await loadHistory(currentPage + 1);
});

clearHistoryBtn.addEventListener("click", async () => {
  const ok = window.confirm("Clear all verification history?");
  if (!ok) return;
  await fetch("/history", { method: "DELETE" });
  await loadHistory(1);
});

loadHistory(1);
