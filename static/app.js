// CST Cloud Regulations RAG Frontend Logic
// SDAIA Agentic AI Program

document.addEventListener("DOMContentLoaded", () => {
    checkSavedKey();
    
    // Enable submitting with Ctrl+Enter or Enter
    const textarea = document.getElementById("queryInput");
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitQuery();
        }
    });
});

function checkSavedKey() {
    const key = localStorage.getItem("cst_gemini_key");
    const statusText = document.getElementById("keyStatusText");
    if (key && key.trim().length > 5) {
        statusText.textContent = "Gemini مفعل ✓";
        statusText.style.color = "#34d399";
    } else {
        statusText.textContent = "مفتاح Gemini API";
        statusText.style.color = "";
    }
}

function openKeyModal() {
    const key = localStorage.getItem("cst_gemini_key") || "";
    document.getElementById("apiKeyInput").value = key;
    document.getElementById("keyModal").classList.remove("hidden");
}

function closeKeyModal() {
    document.getElementById("keyModal").classList.add("hidden");
}

function saveApiKey() {
    const key = document.getElementById("apiKeyInput").value.trim();
    if (key) {
        localStorage.setItem("cst_gemini_key", key);
    } else {
        localStorage.removeItem("cst_gemini_key");
    }
    checkSavedKey();
    closeKeyModal();
}

function toggleReferenceModal() {
    const modal = document.getElementById("referenceModal");
    modal.classList.toggle("hidden");
}

function setQuery(text) {
    const input = document.getElementById("queryInput");
    input.value = text;
    input.focus();
    submitQuery();
}

async function submitQuery() {
    const input = document.getElementById("queryInput");
    const query = input.value.trim();

    if (!query) {
        alert("الرجاء كتابة استفسارك أو اختيار أحد الأمثلة السريعة.");
        return;
    }

    const submitBtn = document.getElementById("submitBtn");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");
    const loading = document.getElementById("loadingIndicator");
    const results = document.getElementById("resultsSection");

    // UI state: loading
    submitBtn.disabled = true;
    btnText.textContent = "جاري التحليل...";
    btnSpinner.classList.remove("hidden");
    loading.classList.remove("hidden");
    results.classList.add("hidden");

    const apiKey = localStorage.getItem("cst_gemini_key") || "";

    try {
        const response = await fetch("/api/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: query,
                api_key: apiKey
            })
        });

        const resData = await response.json();

        if (response.ok && resData.status === "success") {
            renderResults(resData.data);
        } else {
            alert(resData.error || resData.message || "حدث خطأ أثناء معالجة الطلب.");
        }
    } catch (err) {
        console.error(err);
        alert("تعذر الاتصال بالخادم. يرجى التأكد من تشغيل التطبيق.");
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = "استعلم عن موقع التخزين";
        btnSpinner.classList.add("hidden");
        loading.classList.add("hidden");
    }
}

function renderResults(data) {
    const results = document.getElementById("resultsSection");
    results.classList.remove("hidden");

    // 1. Storage Location Verdict
    document.getElementById("resStorageLocation").textContent = data.storage_location;
    
    // 2. Generation Mode Badge
    document.getElementById("genModeBadge").textContent = `وضع التحليل: ${data.generation_mode}`;

    // 3. Classification Badge
    const classBadge = document.getElementById("resClassificationBadge");
    classBadge.textContent = data.classification;
    classBadge.className = "badge"; // reset

    if (data.classification.includes("سرية للغاية") || data.classification.includes("سري للغاية")) {
        classBadge.classList.add("badge-danger");
    } else if (data.classification.includes("سري")) {
        classBadge.classList.add("badge-warning");
    } else if (data.classification.includes("مقيّد") || data.classification.includes("مقيد")) {
        classBadge.classList.add("badge-yellow");
    } else if (data.classification.includes("عام")) {
        classBadge.classList.add("badge-success");
    } else {
        classBadge.classList.add("badge-info");
    }

    // 4. Provider Class
    document.getElementById("resProviderClass").textContent = data.provider_class;

    // 5. Cross Border Policy
    document.getElementById("resCrossBorder").textContent = data.cross_border_policy;

    // 6. Approval Required
    document.getElementById("resApproval").textContent = data.approval_required;

    // 7. Explanation Formatted Text
    const expContainer = document.getElementById("explanationText");
    expContainer.innerHTML = parseSimpleMarkdown(data.answer);

    // 8. Citations / Retrieved Chunks
    const chunksList = document.getElementById("chunksList");
    chunksList.innerHTML = "";
    document.getElementById("chunkCount").textContent = data.retrieved_chunks.length;

    data.retrieved_chunks.forEach((chunk, index) => {
        const card = document.createElement("div");
        card.className = "chunk-card";
        card.innerHTML = `
            <div class="chunk-header">
                <span class="chunk-title">[مقطع ${index + 1}] ${chunk.title}</span>
                <span class="chunk-badge">صفحة ${chunk.page} | ${chunk.section_id} | تطابق: ${(chunk.score * 100).toFixed(0)}%</span>
            </div>
            <div class="chunk-content">${chunk.content}</div>
        `;
        chunksList.appendChild(card);
    });

    // 9. Articles List
    const articlesList = document.getElementById("articlesList");
    articlesList.innerHTML = "";
    data.legal_articles.forEach(art => {
        const li = document.createElement("li");
        li.className = "article-item";
        li.innerHTML = `
            <span class="article-icon">⚖️</span>
            <span class="article-name">${art}</span>
        `;
        articlesList.appendChild(li);
    });

    // Smooth scroll down to results
    results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function switchTab(tabId) {
    const tabs = document.querySelectorAll(".tab-btn");
    const contents = document.querySelectorAll(".tab-content");

    tabs.forEach(t => t.classList.remove("active"));
    contents.forEach(c => c.classList.remove("active"));

    if (tabId === "explanation") {
        tabs[0].classList.add("active");
        document.getElementById("tabExplanation").classList.add("active");
    } else if (tabId === "citations") {
        tabs[1].classList.add("active");
        document.getElementById("tabCitations").classList.add("active");
    } else if (tabId === "articles") {
        tabs[2].classList.add("active");
        document.getElementById("tabArticles").classList.add("active");
    }
}

// Simple Markdown to HTML parser for rendering responses
function parseSimpleMarkdown(text) {
    if (!text) return "";
    let html = text
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/gim, '<em>$1</em>')
        .replace(/^\* (.*$)/gim, '<li>$1</li>')
        .replace(/^\- (.*$)/gim, '<li>$1</li>')
        .replace(/\n\n/gim, '<p></p>');

    // Wrap list items in <ul>
    html = html.replace(/(<li>[\s\S]*?<\/li>)/gm, '<ul>$1</ul>');
    // Remove repeated <ul> inside <ul>
    html = html.replace(/<\/ul>\s*<ul>/g, '');
    
    return html;
}
