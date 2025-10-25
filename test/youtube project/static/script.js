console.log("YouTube Comment Analyzer script loaded!");

document.getElementById("analyzeBtn").addEventListener("click", analyze);

async function analyze() {
    const link = document.getElementById("ytLink").value.trim();
    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = "🔍 Processing... please wait.";

    try {
        const res = await axios.post("/analyze", { link });
        const data = res.data;

        if (data.error) {
            resultDiv.innerHTML = "❌ " + data.error;
            return;
        }

        const o = data.overview;
        resultDiv.innerHTML = `
            <h3>${o.video_title} (${o.video_channel})</h3>
            <p>Total comments: ${o.total_comments}</p>
            <ul>
                <li>😊 Positive: ${o.positive}</li>
                <li>😐 Neutral: ${o.neutral}</li>
                <li>😡 Negative: ${o.negative}</li>
            </ul>
            <h4>Sample comments:</h4>
            <ul>${data.examples.map(c => `<li>[${c.label}] ${c.comment}</li>`).join('')}</ul>
        `;
    } catch (err) {
        console.error(err);
        resultDiv.innerHTML = "⚠️ Error: " + err.message;
    }
}
