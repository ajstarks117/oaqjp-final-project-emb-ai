"""Flask server for the emotion detector application."""

from __future__ import annotations

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.errorhandler(400)
def bad_request(_error):
    """Render the form with a visible message on invalid requests."""
    message = "Bad Request: Please enter some text to analyze."
    return render_template(
        "index.html",
        error=message,
        result=None,
    ), 400


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the page and analyze submitted text."""
    text_to_analyze = ""
    result = None
    error = None

    if request.method == "POST":
        text_to_analyze = request.form.get("text", "")
        if not text_to_analyze or not text_to_analyze.strip():
            return bad_request(None)

        try:
            result = emotion_detector(text_to_analyze)
        except ValueError as exc:
            error = str(exc)

    return render_template("index.html", text=text_to_analyze, result=result, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
