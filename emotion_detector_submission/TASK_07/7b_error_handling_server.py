"""Flask server with blank-input validation."""

from flask import Flask, render_template, request
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)

@app.errorhandler(400)
def handle_bad_request(_error):
    return render_template("index.html", error="Bad Request: Please enter some text to analyze.", result=None), 400

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text_to_analyze = request.form.get("text", "")
        if not text_to_analyze or not text_to_analyze.strip():
            return handle_bad_request(None)
        result = emotion_detector(text_to_analyze)
        return render_template("index.html", result=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
