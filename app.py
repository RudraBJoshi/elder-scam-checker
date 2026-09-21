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

from flask import Flask, render_template, request, jsonify, g
from werkzeug.middleware.proxy_fix import ProxyFix

from model.scam_detector import ScamDetector
from usage_logger import hash_ip, log_usage

BASE_DIR = pathlib.Path(__file__).parent
MAX_MESSAGE_LENGTH = 3000  # matches the dataset's text cap (see project README)
MAX_SENDER_LENGTH = 200

app = Flask(__name__)
detector = ScamDetector(model_dir=BASE_DIR / "model")

# When deployed behind a reverse proxy (Render, PythonAnywhere, etc.), the
# real visitor's IP arrives in the X-Forwarded-For header, not the raw
# socket connection Flask sees by default -- ProxyFix teaches Flask to read
# it from there. Only turn this on once you've actually confirmed you're
# behind a proxy you trust: blindly trusting X-Forwarded-For with no real
# proxy in front of you means anyone can fake any "session" just by setting
# that header themselves. Set BEHIND_PROXY=1 in the hosting platform's
# environment variables when deploying; leave it unset for local dev.
if os.environ.get("BEHIND_PROXY") == "1":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)


@app.before_request
def _compute_session_hash():
    # See usage_logger.py for why this is a salted hash and not the raw IP.
    g.session_hash = hash_ip(request.remote_addr)


@app.route("/")
def index():
    log_usage("/", g.session_hash, outcome="page_view")
    return render_template("index.html")


@app.route("/about")
def about():
    log_usage("/about", g.session_hash, outcome="page_view")
    return render_template("about.html", metrics=detector.metrics)


@app.route("/check", methods=["POST"])
def check():
    # Accept both JSON (fetch) and form-encoded (no-JS fallback) submissions.
    if request.is_json:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            payload = {}
        text = payload.get("message", "")
        sender = payload.get("sender", "")
    else:
        text = request.form.get("message", "")
        sender = request.form.get("sender", "")

    text = text[:MAX_MESSAGE_LENGTH] if isinstance(text, str) else ""
    sender = sender[:MAX_SENDER_LENGTH] if isinstance(sender, str) else ""
    result = detector.analyze(text, sender=sender)

    wants_json = request.is_json or request.headers.get("X-Requested-With") == "fetch"

    if result is None:
        log_usage("/check", g.session_hash, outcome="empty_message")
        if wants_json:
            return jsonify({"error": "empty_message"}), 400
        return render_template("index.html", error="Please paste or type a message to check.")

    log_usage("/check", g.session_hash, risk_level=result["risk_level"], outcome="ok")

    if wants_json:
        return jsonify(result)

    return render_template("index.html", result=result, submitted_text=text, submitted_sender=sender)


if __name__ == "__main__":
    # macOS binds AirPlay Receiver to port 5000 by default, which collides with
    # Flask's default port. Override with PORT=5001 python3 app.py if needed,
    # or turn off AirPlay Receiver in System Settings -> General -> AirDrop & Handoff.
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
