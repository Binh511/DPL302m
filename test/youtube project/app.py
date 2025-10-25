from flask import Flask, render_template, request, jsonify
from youtube_scraper.youtube_scraper import YouTubeCommentScraper
import re
import os

# ==========================
# Flask setup
# ==========================
app = Flask(__name__)

# YouTube API key (thay bằng key của bạn)
YOUTUBE_API_KEY = "AIzaSyBmZdHO7wWwUis6ZC4G9zV4inCU8rcGYwE"

# ==========================
# Simple Sentiment Classifier
# ==========================
def simple_rule_classifier(comment: str) -> str:
    """Basic rule-based sentiment classification"""
    text = comment.lower()
    pos = ["good", "great", "love", "amazing", "helpful", "useful", "thanks", "nice", "cool"]
    neg = ["bad", "terrible", "hate", "disagree", "worst", "awful", "poor"]
    for w in pos:
        if w in text:
            return "positive"
    for w in neg:
        if w in text:
            return "negative"
    spam = ["buy followers", "subscribe for", "visit my channel"]
    for w in spam:
        if w in text:
            return "negative"
    return "neutral"

# ==========================
# Routes
# ==========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    link = data.get("link", "").strip()
    if not link:
        return jsonify({"error": "No link provided"}), 400

    try:
        # 1️⃣ Khởi tạo scraper
        scraper = YouTubeCommentScraper(YOUTUBE_API_KEY)

        # 2️⃣ Gọi API lấy comment thật (tối đa 100 cho nhanh)
        comments, video_info, _ = scraper.scrape_comments(link, max_results=100)

        if not comments:
            return jsonify({"error": "No comments found"}), 404

        # 3️⃣ Phân loại cảm xúc từng comment
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        classified = []

        for c in comments:
            lbl = simple_rule_classifier(c["text_original"])
            counts[lbl] += 1
            classified.append({"comment": c["text_original"], "label": lbl})

        # 4️⃣ Chuẩn bị kết quả
        total = len(comments)
        overview = {
            "total_comments": total,
            "positive": counts["positive"],
            "negative": counts["negative"],
            "neutral": counts["neutral"],
            "video_title": video_info.get("title"),
            "video_channel": video_info.get("channel"),
        }

        return jsonify({"overview": overview, "examples": classified[:20]})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
