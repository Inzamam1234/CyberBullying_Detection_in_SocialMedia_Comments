console.log("🔥 Extension Loaded");

const API_URL = "http://127.0.0.1:8000/predict";

// ===============================
// API CALL
// ===============================
async function analyzeComment(text) {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    return await response.json();
  } catch (error) {
    console.error("API Error:", error);
    return null;
  }
}

// ===============================
// BLUR FUNCTION (CLEAN)
// ===============================
function blurComment(el) {
  el.style.filter = "blur(5px)";
  el.style.transition = "0.3s";
}

// ===============================
// MAIN SCAN FUNCTION
// ===============================
async function scanPage() {
  console.log("Scanning page...");

  const elements = document.querySelectorAll(".text");

  for (let el of elements) {
    if (el.dataset.checked) continue;

    const text = el.innerText.trim();

    if (!text || text.length < 5) continue;

    el.dataset.checked = "true";

    console.log("Checking:", text);

    const result = await analyzeComment(text);

    console.log("Result:", result);

    if (!result) continue;

    // ✅ KEY FIX
    if (result.is_toxic === true) {
      blurComment(el);

      // update count
      chrome.storage.local.get(["hiddenCount"], (data) => {
        let count = data.hiddenCount || 0;
        chrome.storage.local.set({ hiddenCount: count + 1 });
      });
    }
  }
}

// ===============================
// AUTO RUN
// ===============================
window.addEventListener("load", () => {
  setTimeout(scanPage, 2000);
});

// ===============================
// DYNAMIC CONTENT (IMPORTANT)
// ===============================
const observer = new MutationObserver(() => {
  scanPage();
});

observer.observe(document.body, {
  childList: true,
  subtree: true
});