async function analyzeComment(text) {
    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        return await response.json();
    } catch (error) {
        console.error("API Error:", error);
        return null;
    }
}


// Example: scan all paragraphs (you can customize later)
async function scanPage() {
    // Only target comment blocks
    const comments = document.querySelectorAll(".comment");

    for (let el of comments) {
        const text = el.innerText;

        if (!text || text.length < 5) continue;

        const result = await analyzeComment(text);

        if (result && result.is_toxic) {
            el.style.backgroundColor = "#ffdddd";
            el.style.color = "#900";
            el.innerText = "🚫 Hidden toxic content";
        }
    }
}

// Run after page loads
setTimeout(scanPage, 3000);