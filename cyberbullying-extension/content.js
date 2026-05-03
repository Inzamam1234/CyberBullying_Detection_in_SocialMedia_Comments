// content.js — Production Chrome Extension
// Works on YouTube, Twitter/X, Reddit, Instagram

// Change this to your deployed API URL after deployment
// For local testing: "http://127.0.0.1:8000"
const API_URL = "http://127.0.0.1:8000";

// ── Site-specific comment selectors ──────────────────────────
const SITE_SELECTORS = {
  "youtube.com": "yt-formatted-string#content-text",
  "twitter.com": '[data-testid="tweetText"]',
  "x.com": '[data-testid="tweetText"]',
  "reddit.com": '[data-testid="comment"] p, .Comment p',
  "instagram.com": "span._aacl._aaco._aacu._aacx._aad7._aade",
  "facebook.com": '[data-ad-rendering-role="body-text"]',
};

// ── Get selector for current site ────────────────────────────
function getSelectorForSite() {
  const hostname = window.location.hostname.replace("www.", "");
  for (const [site, selector] of Object.entries(SITE_SELECTORS)) {
    if (hostname.includes(site)) return selector;
  }
  return null;
}

// ── Blur a toxic element ──────────────────────────────────────
function blurElement(el, labels) {
  if (el.dataset.toxicProcessed) return;
  el.dataset.toxicProcessed = "true";

  const original = el.innerHTML;

  // Create blurred overlay
  const wrapper = document.createElement("span");
  wrapper.className = "toxic-filter-blur";
  wrapper.innerHTML = original;
  wrapper.style.cssText = `
    filter: blur(6px);
    cursor: pointer;
    user-select: none;
    display: inline;
  `;

  // Tooltip showing why it was flagged
  const tooltip = document.createElement("span");
  tooltip.textContent = `⚠️ Flagged: ${labels.join(", ")} — click to reveal`;
  tooltip.style.cssText = `
    display: block;
    font-size: 11px;
    color: #888;
    margin-bottom: 2px;
  `;

  // Click to reveal
  wrapper.addEventListener("click", () => {
    wrapper.style.filter = "none";
    wrapper.style.cursor = "default";
    tooltip.style.display = "none";
  });

  el.innerHTML = "";
  el.appendChild(tooltip);
  el.appendChild(wrapper);
}

// ── Batch API call ─────────────────────────────────────────────
async function checkBatch(elements) {
  const texts = elements.map((el) => el.innerText.trim());

  try {
    const response = await fetch(`${API_URL}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts }),
    });

    if (!response.ok) return null;
    const data = await response.json();
    return data.results;
  } catch (err) {
    // API not reachable — fail silently, don't break the page
    return null;
  }
}

// ── Process elements in chunks to avoid overloading API ───────
async function processInChunks(elements, chunkSize = 10) {
  for (let i = 0; i < elements.length; i += chunkSize) {
    const chunk = elements.slice(i, i + chunkSize);
    const results = await checkBatch(chunk);

    if (!results) continue;

    results.forEach((result, idx) => {
      if (result.is_toxic) {
        blurElement(chunk[idx], result.labels);
      }
    });

    // Small delay between chunks to avoid hammering the API
    if (i + chunkSize < elements.length) {
      await new Promise((r) => setTimeout(r, 300));
    }
  }
}

// ── Main scan function ─────────────────────────────────────────
async function scanPage() {
  const selector = getSelectorForSite();
  if (!selector) return;

  const allElements = Array.from(document.querySelectorAll(selector));

  // Only process unprocessed elements with meaningful text
  const unprocessed = allElements.filter(
    (el) =>
      !el.dataset.toxicProcessed &&
      el.innerText.trim().length > 10 &&
      el.innerText.trim().length < 2000
  );

  if (unprocessed.length === 0) return;

  console.log(`[ToxicFilter] Scanning ${unprocessed.length} comments...`);
  await processInChunks(unprocessed);
}

// ── Initial scan ──────────────────────────────────────────────
window.addEventListener("load", () => {
  setTimeout(scanPage, 2500);
});

// ── Watch for new comments (infinite scroll / live updates) ───
let scanTimeout = null;
const observer = new MutationObserver(() => {
  // Debounce — don't scan on every tiny DOM change
  clearTimeout(scanTimeout);
  scanTimeout = setTimeout(scanPage, 1500);
});

observer.observe(document.body, {
  childList: true,
  subtree: true,
});