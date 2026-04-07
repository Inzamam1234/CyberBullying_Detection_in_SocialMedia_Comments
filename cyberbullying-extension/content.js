console.log("🔥 Extension Loaded");

async function checkToxicity(inputText) {
    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: inputText })
        });

        return await response.json();

    } catch (err) {
        console.log("API Error:", err);
        return null;
    }
}

async function scanPage() {
    console.log("Scanning page...");

    const elements = document.querySelectorAll(".text");

    for (let i = 0; i < elements.length; i++) {

        const el = elements[i];

        const content = el.innerText;

        console.log("Checking:", content);

        if (content && content.length > 15 && content.length < 200) {

            const result = await checkToxicity(content);

            console.log("Result:", result);

            if (
                result &&
                (result.toxic > 0.3 || result.insult > 0.3 || result.obscene > 0.3)
            ) {
                const span = document.createElement("span");
                span.innerText = el.innerText;

                span.style.filter = "blur(4px)";
                span.style.opacity = "0.7";

                el.innerText = "";
                el.appendChild(span);
            }
        }
    }
}

window.addEventListener("load", () => {
    setTimeout(scanPage, 4000);
});