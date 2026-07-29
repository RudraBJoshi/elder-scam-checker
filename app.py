"""
Elder Scam Detector -- Flask app.

A senior pastes or types a suspicious text/email/call description into a big,
simple text box and gets back a plain-language risk assessment: a traffic-light
verdict, why it was flagged (in everyday language, not ML jargon), and what to
do next. Built for the 2026 Congressional App Challenge (CA-50).

Routes:
    GET  /        - the checker page (textarea + button)
    POST /check   - analyze submitted text; returns JSON for the fetch-based
                    UI, and also works as a plain form POST fallback (renders
                    the result server-side) if JavaScript is unavailable.
    GET  /about   - plain-language "how this works" page for the demo/judges
"""
import os
import pathlib

from flask import Flask, render_template, request, jsonify

from model.scam_detector import ScamDetector

BASE_DIR = pathlib.Path(__file__).parent
MAX_MESSAGE_LENGTH = 3000  # matches the dataset's text cap (see project README)
MAX_SENDER_LENGTH = 200

app = Flask(__name__)
detector = ScamDetector(model_dir=BASE_DIR / "model")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html", metrics=detector.metrics)


@app.route("/check", methods=["POST"])
def check():
    # Accept both JSON (fetch) and form-encoded (no-JS fallback) submissions.
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        text = payload.get("message", "")
        sender = payload.get("sender", "")
    else:
        text = request.form.get("message", "")
        sender = request.form.get("sender", "")

    text = text[:MAX_MESSAGE_LENGTH]
    sender = (sender or "")[:MAX_SENDER_LENGTH]
    result = detector.analyze(text, sender=sender)

    wants_json = request.is_json or request.headers.get("X-Requested-With") == "fetch"

    if result is None:
        if wants_json:
            return jsonify({"error": "empty_message"}), 400
        return render_template("index.html", error="Please paste or type a message to check.")

    if wants_json:
        return jsonify(result)

    return render_template("index.html", result=result, submitted_text=text, submitted_sender=sender)


if __name__ == "__main__":
    # macOS binds AirPlay Receiver to port 5000 by default, which collides with
    # Flask's default port. Override with PORT=5001 python3 app.py if needed,
    # or turn off AirPlay Receiver in System Settings -> General -> AirDrop & Handoff.
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
